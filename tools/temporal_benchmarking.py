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
    def __init__(self, onnx_path, fp16=True, workspace_size_gb=4):
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


def trt_benchmark(onnx_filename, csv_filename = f"trt_benchmark.csv", fp16=True):
    """
    Convenience function to benchmark an ONNX model using TensorRT.
    Builds the engine, warms up, runs timing, and prints latency.
    """
    bench = TRTBenchmark(onnx_filename, fp16=fp16)
    bench.build_engine()
    bench.warmup(iterations=10)
    avg, p95 = bench.benchmark(runs=100)
    ips = 1000 / avg
    bench.profile_layers(top_n=100, profile_file=csv_filename)
    print(f"Average latency: {avg:.3f} ms")
    print(f"p95 latency: {p95:.3f} ms")
    print(f"Iterations per second: {ips:.2f} inferences/sec.")
    return avg, p95, ips
