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
        self.memory_snapshots = []
        # Capture baseline memory before any TensorRT operations
        self.memory_snapshots.append(('before_engine_creation', self.get_gpu_memory_info()))

    def build_engine(self):
        builder = trt.Builder(self.logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)

        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                # Enhanced error reporting
                error_msg = "Failed to parse ONNX model. Detailed errors:\n"
                num_errors = parser.num_errors
                error_msg += f"Number of errors: {num_errors}\n"
                
                for i in range(num_errors):
                    error = parser.get_error(i)
                    error_msg += f"  Error {i+1}:\n"
                    error_msg += f"    Code: {error.code()}\n"
                    error_msg += f"    Description: {error.desc()}\n"
                    error_msg += f"    File: {error.file()}\n"
                    error_msg += f"    Line: {error.line()}\n"
                    error_msg += f"    Function: {error.func()}\n"
                    error_msg += f"    Node: {error.node()}\n"
                
                # Print to stderr for immediate visibility in logs
                # import sys
                print(error_msg)
                
                raise RuntimeError(error_msg)

        # with open(self.onnx_path, "rb") as f:
        #     if not parser.parse(f.read()):
        #         raise RuntimeError("Failed to parse ONNX model")

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

        # Capture memory after engine creation
        self.memory_snapshots.append(('after_engine_creation', self.get_gpu_memory_info()))

    def get_gpu_memory_info(self):
        """Get current GPU memory usage."""
        import pycuda.driver as cuda
        
        free_mem, total_mem = cuda.mem_get_info()
        used_mem = total_mem - free_mem
        
        return {
            'total_mb': total_mem / (1024**2),
            'used_mb': used_mem / (1024**2),
            'free_mb': free_mem / (1024**2),
            'used_percent': (used_mem / total_mem) * 100
        }

    def get_engine_memory_footprint(self):
        """Get TensorRT engine memory requirements."""
        if self.engine is None:
            return None
        
        # Calculate memory for weights and activations
        engine_size = 0
        activation_memory = 0
        
        # Engine serialization size (weights)
        serialized_engine = self.engine.serialize()
        engine_size = len(bytes(serialized_engine)) / (1024**2)  # MB
        # engine_size = serialized_engine.size / (1024**2)  # MB
        
        # Calculate activation memory from bindings
        for i in range(self.engine.num_bindings):
            shape = self.context.get_binding_shape(i)
            dtype = self.engine.get_binding_dtype(i)
            size_bytes = abs(int(np.prod(shape))) * np.dtype(trt.nptype(dtype)).itemsize
            activation_memory += size_bytes
        
        return {
            'engine_weights_mb': engine_size,
            'activation_memory_mb': activation_memory / (1024**2),
            'total_model_memory_mb': engine_size + (activation_memory / (1024**2))
        }

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
        inference_memories = []
        self._prepare_dummy_inputs()
        times = []
        for i in range(runs):
            before_inference = self.get_gpu_memory_info()
        
            start = time.perf_counter()
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            self.stream.synchronize()
            times.append((time.perf_counter() - start) * 1000)

            after_inference = self.get_gpu_memory_info()
            inference_memories.append(after_inference['used_mb'])

            if i == 0:  # Record first run details
                self.memory_snapshots.append(('before_first_inference', before_inference))
                self.memory_snapshots.append(('after_first_inference', after_inference))

        memory_stats = {
            'mem_mb_mean': np.mean(inference_memories),
            'mem_mb_std': np.std(inference_memories),
            'mem_mb_min': np.min(inference_memories),
            'mem_mb_max': np.max(inference_memories)
        }

        avg = np.mean(times)
        p95 = np.percentile(times, 95)
        return avg, p95, memory_stats

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

        # Check if model expects depth input by examining bindings or config
        num_inputs = sum(1 for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i))
        expects_depth_from_bindings = num_inputs == 3  # left, right, depth_src_0
        
        # Also check from model config if available
        # expects_depth_from_config = False
        # if hasattr(self.cfgs, 'MODEL') and hasattr(self.cfgs.MODEL, 'ADDITIONAL_DEPTH_SRC'):
        #     expects_depth_from_config = self.cfgs.MODEL.ADDITIONAL_DEPTH_SRC
        
        # expects_depth = expects_depth_from_bindings or expects_depth_from_config
        expects_depth = expects_depth_from_bindings
        
        print(f"Model expects {num_inputs} inputs. Depth input required: {expects_depth}")
        print(f"  - From bindings: {expects_depth_from_bindings}")
        # print(f"  - From config: {expects_depth_from_config}")

        d_left = None
        d_right = None
        d_depth = None  # Add depth buffer
        d_output = None
        current_left_size = 0
        current_right_size = 0
        current_depth_size = 0  # Add depth size tracking
        current_output_size = 0

        try:
            # Iterate over the PyTorch DataLoader
            for i, data in enumerate(self.data_loader):
                left_input = data['left'].cpu().numpy().astype(np.float32)[0]
                right_input = data['right'].cpu().numpy().astype(np.float32)[0]
                
                # Handle depth input if required
                depth_input = None
                if expects_depth:
                    if 'depth_src_0' in data:
                        depth_input = data['depth_src_0'].cpu().numpy().astype(np.float32)[0]
                    else:
                        print(f"Warning: Model expects depth_src_0 but not found in data. Skipping sample {i}")
                        continue
                
                # Check if we need to allocate or resize buffers
                left_nbytes = left_input.nbytes
                right_nbytes = right_input.nbytes
                depth_nbytes = depth_input.nbytes if depth_input is not None else 0
                
                # Set shape for current input
                # this is useful only if training samples change size but in our case they dont
                # self.context.set_binding_shape(0, (1,) + left_input.shape)  # left input # add batch dimension
                # self.context.set_binding_shape(1, (1,) + right_input.shape)  # right input
                
                output_shape = self.context.get_binding_shape(self.output_binding_index)
                output = np.empty(output_shape, dtype=np.float32)
                output_nbytes = output.nbytes
                
                # CHECK AND RESIZE BUFFERS IF NEEDED
                need_realloc = (
                    d_left is None or left_nbytes > current_left_size or 
                    d_right is None or right_nbytes > current_right_size or
                    d_output is None or output_nbytes > current_output_size or
                    (expects_depth and (d_depth is None or depth_nbytes > current_depth_size))
                )
                
                if need_realloc:
                    # Free existing buffers if they exist
                    if d_left is not None:
                        d_left.free()
                    if d_right is not None:
                        d_right.free()
                    if d_depth is not None:
                        d_depth.free()
                    if d_output is not None:
                        d_output.free()
                    
                    # Allocate new buffers with current sizes
                    print(f"Allocating GPU buffers: left={left_nbytes//1024//1024}MB, "
                        f"right={right_nbytes//1024//1024}MB, "
                        f"depth={depth_nbytes//1024//1024}MB, "
                        f"output={output_nbytes//1024//1024}MB")
                    
                    d_left = cuda.mem_alloc(left_nbytes)
                    d_right = cuda.mem_alloc(right_nbytes)
                    if expects_depth:
                        d_depth = cuda.mem_alloc(depth_nbytes)
                    d_output = cuda.mem_alloc(output_nbytes)
                    
                    # Update current sizes
                    current_left_size = left_nbytes
                    current_right_size = right_nbytes
                    if expects_depth:
                        current_depth_size = depth_nbytes
                    current_output_size = output_nbytes
                
                # Transfer inputs to GPU (reuse existing buffers)
                cuda.memcpy_htod(d_left, left_input)
                cuda.memcpy_htod(d_right, right_input)
                if expects_depth:
                    cuda.memcpy_htod(d_depth, depth_input)

                # Run inference with appropriate bindings
                if expects_depth:
                    bindings = [int(d_left), int(d_right), int(d_depth), int(d_output)]
                else:
                    bindings = [int(d_left), int(d_right), int(d_output)]
                    
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
            
        except Exception as e:
            print(f"Error during inference {i}: {e}.")
            print(f"Returning prematurely with metrics collected so far.")
            
        finally:
            # Free GPU memory
            if d_left is not None:
                d_left.free()
            if d_right is not None:
                d_right.free()
            if d_depth is not None:
                d_depth.free()
            if d_output is not None:
                d_output.free()

        # Compute final averages
        results = {k: float(torch.tensor(epoch_metrics[k]["values"]).mean()) for k in epoch_metrics.keys()}

        # self.logger.info(f"Testing TensorRT: Metrics: {results}")
        print(f"Testing TensorRT: Metrics: {results}")

        return results
    
# Benchmarking using ONNX Runtime
# @torch.no_grad()
def onnx_profile(onnx_path, shape, provider='CUDAExecutionProvider', iters=200, warmup=20):
    import onnxruntime as ort
    import numpy as np, time, json, collections

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # 0=VERBOSE 1=INFO 2=WARNING 3=ERROR 4=FATAL
    sess_opts.enable_profiling = True
    # 2025-11-14 17:51:17.445014573 [V:onnxruntime:, session_state.cc:1146 VerifyEachNodeIsAssignedToAnEp] Node placements
    # 2025-11-14 17:51:17.445030422 [V:onnxruntime:, session_state.cc:1149 VerifyEachNodeIsAssignedToAnEp]  All nodes placed on [CUDAExecutionProvider]. Number of nodes: 519
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Optional: save optimized graph
    sess_opts.optimized_model_filepath = onnx_path.replace(".onnx", ".opt.onnx")

    providers = [(provider, {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"] if provider != "CPUExecutionProvider" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)

    B, C, H, W = shape
    feeds = {}
    inps = sess.get_inputs()
    feeds[inps[0].name] = np.random.randn(B, C, H, W).astype(np.float32)
    feeds[inps[1].name] = np.random.randn(B, C, H, W).astype(np.float32)
    # optional extra inputs (e.g., depth source)
    for i in inps[2:]:
        if 'depth' in i.name:
            feeds[i.name] = np.random.randn(B, H // 2, W // 2).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, feeds)

    t0 = time.time()
    for _ in range(iters):
        sess.run(None, feeds)
    t1 = time.time()
    throughput = iters / (t1 - t0)

    profile_path = sess.end_profiling()
    print(f"ORT profile saved to: {profile_path}")

    # Robust profile parser (handles chrome-trace JSON and JSONL)
    def load_ort_profile(path):
        import json, os
        with open(path, "rb") as f:
            raw = f.read()
        # Strip BOM and NULs, decode
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        txt = raw.replace(b"\x00", b"").decode("utf-8", errors="ignore").strip()
        # Try full JSON first (dict or list)
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                return obj.get("traceEvents", obj.get("events", []))
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
        # Fallback: JSONL (one JSON per line)
        events = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            # strip BOM if present per-line
            if line and line[0] == "\ufeff":
                line = line[1:]
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # skip lines that aren’t JSON records
                continue
        return events

    events = load_ort_profile(profile_path)
    agg_ms = collections.Counter()
    for e in events:
        if e.get("cat") == "Node":  # operator node
            agg_ms[e["name"]] += e["dur"] / 1000.0  # us -> ms
    print("Top ops by total time (ms):")
    for name, ms in agg_ms.most_common(20):
        print(f"{name:40s} {ms:8.3f}")
    return throughput, profile_path

def trt_benchmark(onnx_filename, csv_filename = None, fp16=True, data_loader=None, cfgs=None):
    """
    Convenience function to benchmark an ONNX model using TensorRT.
    Builds the engine, warms up, runs timing, and prints latency.
    """
    bench = TRTBenchmark(onnx_filename, fp16=fp16, data_loader=data_loader, cfgs=cfgs)
    bench.build_engine()
    engine_footprint = bench.get_engine_memory_footprint()
    bench.warmup(iterations=50)
    avg, p95, memory_inference_stats = bench.benchmark(runs=500)
    ips = 1000 / avg
    if csv_filename is not None:
        bench.profile_layers(top_n=100, profile_file=csv_filename)
    print(f"Average latency: {avg:.3f} ms")
    print(f"p95 latency: {p95:.3f} ms")
    print(f"Iterations per second: {ips:.2f} inferences/sec.")
    acc_results = bench.test_on_real_data_trt()
    return avg, p95, ips, acc_results, engine_footprint, memory_inference_stats
