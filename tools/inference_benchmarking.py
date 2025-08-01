import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os


import csv
from collections import defaultdict

class CustomProfiler(trt.IProfiler):
    def __init__(self, save_path="trt_layer_profile.csv", ignore_copies=True):
        super().__init__()
        self.records = []
        self.save_path = save_path
        self.ignore_copies = ignore_copies

    def report_layer_time(self, layer_name, ms):
        if self.ignore_copies and "Reformatting CopyNode" in layer_name:
            return
        self.records.append((layer_name, ms))

    def save_to_csv(self):
        with open(self.save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Layer Name", "Time (ms)"])
            for name, ms in self.records:
                writer.writerow([name, ms])

    def summarize(self, top_n=20):
        sorted_layers = sorted(self.records, key=lambda x: x[1], reverse=True)
        print(f"\nTop {top_n} layers by execution time:")
        for name, ms in sorted_layers[:top_n]:
            print(f"{name:80s} {ms:.4f} ms")

class TRTBenchmark:
    def __init__(self, onnx_path, fp16=True, workspace_size_gb=4, data_loader=None, cfgs=None):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
        self.onnx_path = onnx_path
        self.logger = trt.Logger(trt.Logger.INFO)
        self.fp16 = fp16
        self.workspace_size = workspace_size_gb << 30  # GB → bytes
        self.engine = None
        self.context = None
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()
        self.data_loader = data_loader  # Optional, for inference data preparation
        self.cfgs = cfgs  # Optional, for testing configurations
        self.output_binding_index = None

    def build_engine(self):
        builder = trt.Builder(self.logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)

        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                raise RuntimeError("Failed to parse ONNX model")

        config = builder.create_builder_config()
        config.max_workspace_size = self.workspace_size
        if self.fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        self.engine = builder.build_engine(network, config)
        if self.engine is None:
            raise RuntimeError("Engine build failed")

        self.context = self.engine.create_execution_context()

        # Find the output binding index
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i) == False:  # This is an output
                binding_name = self.engine.get_binding_name(i)
                if binding_name == "disp_pred":  # or whatever your output name is
                    self.output_binding_index = i
                    break
        
        if self.output_binding_index is None:
            raise ValueError("Could not find output binding 'disp_pred' in TensorRT engine")

        self._allocate_buffers()

    def _allocate_buffers(self):
        self.bindings.clear()
        self.inputs.clear()
        self.outputs.clear()
        for i, binding in enumerate(self.engine):
            shape = self.context.get_binding_shape(i)
            size = abs(int(np.prod(shape)))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(i):
                self.inputs.append((binding, shape, device_mem, dtype))
            else:
                self.outputs.append((binding, shape, device_mem, dtype))

    def _prepare_dummy_inputs(self):
        for name, shape, device_mem, dtype in self.inputs:
            h_data = np.random.randn(*shape).astype(dtype)
            cuda.memcpy_htod(device_mem, h_data)

    def warmup(self, iterations=10):
        self._prepare_dummy_inputs()
        for _ in range(iterations):
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            self.stream.synchronize()

    def benchmark(self, runs=100):
        self._prepare_dummy_inputs()
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            self.stream.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        avg = np.mean(times)
        p95 = np.percentile(times, 95)
        return avg, p95

    def profile_layers(self, top_n=100, profile_file="trt_layer_profile.csv"):
        profiler = CustomProfiler(save_path=profile_file)
        self.context.profiler = profiler
        self.context.execute_v2(self.bindings)
        profiler.save_to_csv()
        profiler.summarize(top_n=top_n)
        print(f"[INFO] Full layer timings saved to {profile_file}")

    def test_on_real_data_trt(self):
        import torch

        if self.data_loader is None:
            print("No data loader provided for TensorRT inference testing. Returning None.")
            return None
        
        if not isinstance(self.data_loader, torch.utils.data.DataLoader):
            print("data_loader must be a PyTorch DataLoader instance. Returning None.")
            return None
        
        print("Starting TensorRT inference testing on real data...")

        # import pycuda.driver as cuda
        import numpy as np
        from functools import partial
        from stereo.evaluation.metric_per_image import epe_metric, d1_metric, threshold_metric
        import time

        # Metric functions used to evaluate predictions
        metric_func_dict = {
            'epe': epe_metric,
            'd1_all': d1_metric,
            'thres_1': partial(threshold_metric, threshold=1),
            'thres_2': partial(threshold_metric, threshold=2),
            'thres_3': partial(threshold_metric, threshold=3),
        }

        testing_cfgs = self.cfgs.TESTING
        # local_rank = self.local_rank
        local_rank = 0  # Assuming single GPU for simplicity, adjust as needed

        # Initialize metric storage
        epoch_metrics = {k: {'indexes': [], 'values': []} for k in testing_cfgs.METRIC}

        # Iterate over the PyTorch DataLoader
        for i, data in enumerate(self.data_loader):
            # Move tensors to CPU and convert to numpy for TensorRT
            left = data['left'].cpu().numpy().astype(np.float32)  # shape [N, C, H, W]
            right = data['right'].cpu().numpy().astype(np.float32)

            # Assuming batch size = 1, squeeze batch dim for simplicity
            left_input = left[0]
            right_input = right[0]

            # Allocate GPU buffers (only once per epoch for speed; shown inline for clarity)
            d_left = cuda.mem_alloc(left_input.nbytes)
            d_right = cuda.mem_alloc(right_input.nbytes)

            # Get output shape from the engine binding
            output_binding_idx = self.output_binding_index  # precomputed index for "disp_pred"
            output_shape = self.context.get_binding_shape(output_binding_idx)
            output = np.empty(output_shape, dtype=np.float32)
            d_output = cuda.mem_alloc(output.nbytes)

            # Transfer inputs to GPU
            cuda.memcpy_htod(d_left, left_input)
            cuda.memcpy_htod(d_right, right_input)

            # Prepare bindings in correct order: [left, right, output]
            bindings = [int(d_left), int(d_right), int(d_output)]

            # Run inference with TensorRT
            infer_start = time.time()
            self.context.execute_v2(bindings)
            infer_time = time.time() - infer_start

            # Copy result back to CPU
            cuda.memcpy_dtoh(output, d_output)

            # TensorRT output is numpy, convert to torch tensor for metric computation
            disp_pred = torch.from_numpy(output).to(local_rank)

            # Access ground truth and mask
            disp_gt = data["disp"].to(local_rank)
            mask = (disp_gt < testing_cfgs.MAX_DISP) & (disp_gt > 0)
            if 'occ_mask' in data and testing_cfgs.get('APPLY_OCC_MASK', False):
                mask = mask & ~data['occ_mask'].to(torch.bool)

            # Compute metrics
            for m in testing_cfgs.METRIC:
                metric_func = metric_func_dict[m]
                res = metric_func(disp_pred.squeeze(1), disp_gt, mask)
                epoch_metrics[m]['indexes'].extend(data['index'].tolist())
                epoch_metrics[m]['values'].extend(res.tolist())

            # Logging inference time
            message = f'Testing TensorRT: Iter:{i:>4d}/{len(self.data_loader)} InferTime: {infer_time*1000:.2f}ms'
            # self.logger.info(message)
            print(message)

        # Compute final averages
        results = {k: torch.tensor(epoch_metrics[k]["values"]).mean() for k in epoch_metrics.keys()}

        # self.logger.info(f"Testing TensorRT: Metrics: {results}")
        print(f"Testing TensorRT: Metrics: {results}")

        return results



def trt_benchmark(onnx_filename, csv_filename = f"trt_benchmark.csv", fp16=True, data_loader=None, cfgs=None):
    """
    Convenience function to benchmark an ONNX model using TensorRT.
    Builds the engine, warms up, runs timing, and prints latency.
    """
    bench = TRTBenchmark(onnx_filename, fp16=fp16, data_loader=data_loader, cfgs=cfgs)
    bench.build_engine()
    bench.warmup(iterations=10)
    avg, p95 = bench.benchmark(runs=100)
    ips = 1000 / avg
    bench.profile_layers(top_n=100, profile_file=csv_filename)
    print(f"Average latency: {avg:.3f} ms")
    print(f"p95 latency: {p95:.3f} ms")
    print(f"Iterations per second: {ips:.2f} inferences/sec.")
    results = bench.test_on_real_data_trt()
    return avg, p95, ips, results
