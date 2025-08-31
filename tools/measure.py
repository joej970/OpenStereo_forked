# @Time    : 2024/3/1 11:17
# @Author  : zhangchenming
import time
from sympy import shape
import torch
import argparse
import sys
import os
import onnx
import thop
from easydict import EasyDict
from tqdm import tqdm

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.modeling import build_trainer


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)
    args.run_mode = 'measure'
    return args, cfgs


def main():
    args, cfgs = parse_config()
    model = build_trainer(args, cfgs, local_rank=0, global_rank=0, logger=None, tb_writer=None).model

    shape = [1, 3, 544, 960]
    infer_time(model, shape)
    measure(model, shape)


@torch.no_grad()
def measure(model, shape):
    model.eval()
    import copy
    # Create a copy of the model for FLOP calculation to avoid hook conflicts
    model_copy = copy.deepcopy(model)

    left_input = torch.randn(shape).cuda()
    right_input = torch.randn(shape).cuda()
    inputs_arr = [left_input, right_input]
    # check if the model has attribute 'additional_depth_src' and if it is true:
    if hasattr(model, 'additional_depth_src') and model.additional_depth_src:
        print("Generationg additional depth source")
        #depth spatial dims are half of original
        depth_shape = [shape[0], shape[2] // 2, shape[3] // 2]
        depth_src_0_input = torch.randn(depth_shape).cuda()
        print(f"Depth source shape: {depth_src_0_input.shape}")
        inputs_arr.append(depth_src_0_input)
        # flops, params = thop.profile(model_copy, inputs=(left_input, right_input, depth_src_0_input))
    else:
        print("No additional depth source")
        # flops, params = thop.profile(model_copy, inputs=(left_input, right_input))

    # print(f"Members of inputs_arr: {len(inputs_arr)}")
    # for i, input_tensor in enumerate(inputs_arr):
    #     print(f"Shape of input tensor {i}: {input_tensor.shape}")

    # print(f"tuple of inputs_arr: {len(tuple(inputs_arr))}")
    # for i, input_tensor in enumerate(tuple(inputs_arr)):
    #     print(f"Shape of input tensor {i} (as tuple): {input_tensor.shape}")

    flops, params = thop.profile(model_copy, inputs=tuple(inputs_arr))

    # Clear the copy to free memory
    del model_copy
    # flops, params = thop.profile(model, inputs=(left_input,right_input))
    message_1 = f"Number of calculates: {flops / 1e9:.2f} GFlops"
    message_2 = f"Number of parameters: {params / 1e6:.2f} M"
    print(message_1)
    print(message_2)
    return message_1, message_2, flops, params

@torch.no_grad()
def infer_time_torch_script(model, shape):
   # CONFIGURE THESE
    batch_size = shape[0]
    # input_shape = (3, 224, 224)
    num_iters = 1000
    warmup_iters = 100
    device = torch.device("cuda")

    # Step 1: Load your model
    # model.eval().to(device)

    # Step 2: Create dummy input
    left_input = torch.randn(shape).cuda()
    right_input = torch.randn(shape).cuda()
    inputs_arr = [left_input, right_input]

    if hasattr(model, 'additional_depth_src') and model.additional_depth_src:
        #depth spatial dims are half of original
        depth_shape = [shape[0], shape[2] // 2, shape[3] // 2]
        depth_src_0_input = torch.randn(depth_shape).cuda()

        inputs_arr.append(depth_src_0_input)
        # Step 3: Trace the model with TorchScript
        # scripted_model = torch.jit.trace(model, (left_input, right_input, depth_src_0_input), strict=False) 
    # else:
        # scripted_model = torch.jit.trace(model, (left_input, right_input), strict=False) 

    scripted_model = torch.jit.trace(model, tuple(inputs_arr), strict=False) 
        # Step 3: Trace the model with TorchScript
    
    # scripted_model = torch.jit.script(model)  # Use script instead of trace
    # Note: If your model has dynamic behavior (like conditionals based on inputs), use script instead of trace.
    # the model returns a dict which is not supported by torch.jit.trace so we need to pass strict=False
    scripted_model.eval().to(device)

    # Step 4: Warm-up (important)
    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = scripted_model(*inputs_arr)
    torch.cuda.synchronize()

    # Step 5: Measure inference time
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = scripted_model(*inputs_arr)
    torch.cuda.synchronize()
    end = time.time()

    # Step 6: Compute throughput
    total_time = end - start
    throughput = batch_size * num_iters / total_time
    message = f"Throughput TorchScript: {throughput:.2f} samples/sec"
    print(message)
    
    return message, throughput


@torch.no_grad()
def infer_time(model, shape):
    model.eval()
    repetitions = 100

    left_input = torch.randn(shape).cuda()
    right_input = torch.randn(shape).cuda()
    inputs_arr = [left_input, right_input]

    if hasattr(model, 'additional_depth_src') and model.additional_depth_src:
        depth_shape = [shape[0], shape[2] // 2, shape[3] // 2]
        depth_src_0 = torch.randn(depth_shape).cuda()
        inputs_arr.append(depth_src_0)

    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(10):
            _ = model(*inputs_arr)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    # torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    # timings = np.zeros((repetitions, 1))

    all_time = 0
    print('testing ...\n')
    with torch.no_grad():
        for _ in tqdm(range(repetitions)):
            # starter.record()
            infer_start = time.perf_counter()
            # infer_start = time.time()
            result = model(*inputs_arr)
            # print(result.keys())
            # ender.record()
            all_time += time.perf_counter() - infer_start
            # torch.cuda.synchronize()  # 等待GPU任务完成

            # curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
            # timings[rep] = curr_time
    t_per_inference = all_time / repetitions * 1000
    throughput = repetitions / all_time
    message_1 = f"Performance ({repetitions} repetitions): {all_time:.3f} seconds, {t_per_inference:.3f} ms per inference, throughput: {throughput:.2f} inferences/sec"
    print(message_1)

    # Benchmark
    num_iters = 1000
    batch_size = shape[0]
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(*inputs_arr)
    torch.cuda.synchronize()
    end = time.time()

    throughput = batch_size * num_iters / (end - start)
    message_2 = f"Throughput: {throughput:.2f} samples/sec"
    print(message_2)
    
    return message_1, message_2, t_per_inference, throughput

    # avg = timings.sum() / repetitions
    # print('\navg_time=%.3fms\n' % avg)
    

def export_to_onnx(model, input_shape, onnx_path,
                   
                   use_fp16=False,
                   dynamic_batch=False,
                   opset_version=17):
    """
    Export a PyTorch model to ONNX.

    Args:
        model (nn.Module): The PyTorch model.
        input_shape (tuple): Input shape excluding batch (e.g., (3, 224, 224)).
        onnx_path (str): Where to save the ONNX file.
        batch_size (int): Batch size to use during tracing.
        use_fp16 (bool): Convert model to half precision before export.
        dynamic_batch (bool): Use dynamic batch size in ONNX.
        opset_version (int): ONNX opset version.
    """
    
    batch_size = input_shape[0]
    
    model = model.eval().cuda()
    if use_fp16:
        model = model.half()

    dtype = torch.float16 if use_fp16 else torch.float32
    dummy_left = torch.randn(*input_shape, dtype=dtype).cuda()
    dummy_right = torch.randn(*input_shape, dtype=dtype).cuda()
    inputs_arr = [dummy_left, dummy_right]

    input_names = ["left", "right"]
    output_names = ["disp_pred"]  # adjust this if you return more

    if hasattr(model, 'additional_depth_src') and model.additional_depth_src:
        depth_shape = [input_shape[0], input_shape[2] // 2, input_shape[3] // 2]
        dummy_depth = torch.randn(*depth_shape, dtype=dtype).cuda()
        inputs_arr.append(dummy_depth)
        input_names.append("depth_src_0")

    if dynamic_batch:
        dynamic_axes = {}
        for i, name in enumerate(input_names):
            dynamic_axes[name] = {0: "batch_size"}
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}
    else:
        dynamic_axes = None

    torch.onnx.export(
        model,
        tuple(inputs_arr),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL
    )

    if not os.path.exists(onnx_path):
        raise RuntimeError(f"ONNX export failed: file not created at {onnx_path}")

    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"Exported ONNX model saved to {onnx_path}")
    except Exception as e:
        raise RuntimeError(f"ONNX model validation failed: {e}")

def format_dict_multiline(d, indent=0):
        """
        Format a dictionary with one key-value pair per line.
        
        Args:
            d (dict): Dictionary to format
            indent (int): Current indentation level
        
        Returns:
            str: Formatted string representation
        """
        lines = []
        indent_str = "  " * indent
        
        if isinstance(d, dict):
            lines.append("{")
            for key, value in d.items():
                if isinstance(value, dict):
                    lines.append(f"{indent_str}  '{key}': {format_dict_multiline(value, indent + 1)}")
                elif isinstance(value, (list, tuple)):
                    lines.append(f"{indent_str}  '{key}': {value}")
                else:
                    lines.append(f"{indent_str}  '{key}': {value}")
            lines.append(f"{indent_str}}}")
        else:
            return str(d)
        
        return "\n".join(lines)

if __name__ == '__main__':
    main()
