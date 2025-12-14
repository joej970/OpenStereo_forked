import os
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from tools/ to project root
sys.path.insert(0, project_root)

from stereo.utils import common_utils
from stereo.datasets import build_dataloader
from easydict import EasyDict
import measure

import config_parsing

def build_test_loader(args, cfgs):
    test_set, test_loader, test_sampler = build_dataloader(
        data_cfg=cfgs.DATA_CONFIG,
        batch_size=1,  # Batch size is usually set to 1 for evaluation
        is_dist=args.dist_mode,
        workers=args.workers,
        pin_memory=args.pin_memory,
        mode='testing')
    # logger.info('Total samples for test dataset: %d' % (len(test_set)))
    return test_set, test_loader, test_sampler

def run_trt_benchmark(onnx_file, args, cfgs):

    import inference_benchmarking
    
    experiment_summary = {}

    local_rank = 0

    log_file = os.path.join(args.output_dir, f"{args.slurm_job_id}_{args.experiment_id}.log")

    # log_file = os.path.join(args.output_dir, 'tensorRT_eval_{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), local_rank))
    logger = common_utils.create_logger(log_file, rank=local_rank)

    test_set, test_loader, test_sampler = build_test_loader(args, cfgs)
    logger.info('Total samples for test dataset: %d' % (len(test_set)))

    print(f"Running TensorRT benchmark on ONNX file: {onnx_file}")
    logger.info(f"Running TensorRT benchmark on ONNX file: {onnx_file}")
    # csv_filename = f"trt_benchmark_{args.slurm_job_id}.csv"
    csv_filename = os.path.join(args.output_dir, f"trt_benchmark.csv")
    curr_csv_filename = csv_filename

    # idx = 0
    experiment_summary = {}

    runs = args.onnx_eval_runs if hasattr(args, 'onnx_eval_runs') else 3
    for run_idx in range(runs):
    # while os.path.exists(onnx_file):
        avg, p95, ips, metrics, engine_footprint, memory_inference_stats = inference_benchmarking.trt_benchmark(onnx_file, csv_filename=curr_csv_filename, fp16=True, data_loader=test_loader, cfgs=cfgs)

        curr_csv_filename = None

        summary = {}
        summary["inference_benchmark"] = {}
        summary["test_accuracy_metrics_trt"] = {}
        summary["memory"] = {}

        summary["inference_benchmark"]["Onnx_trt_inference"] = {
            "avg_latency": avg,
            "p95_latency": p95,
            "inference": ips
        }
        summary["test_accuracy_metrics_trt"] = metrics
        summary["memory"] = {
            "engine_footprint": engine_footprint,
            "inference_memory_stats": memory_inference_stats
        }

        experiment_summary[f"run_{run_idx:02d}"] = summary
        # idx += 1
        # onnx_file = f"{onnx_file[:-8]}_run{idx}.onnx"

        print(f"Completed TensorRT benchmark run {run_idx}/{runs-1}: summary: {summary}")

        # experiment_summary["inference_benchmark"] = {}
        # experiment_summary["test_accuracy_metrics_trt"] = {}
        # experiment_summary["memory"] = {}

        # experiment_summary["inference_benchmark"]["Onnx_trt_inference"] = {
        #     "avg_latency": avg,
        #     "p95_latency": p95,
        #     "inference": ips
        # }
        # experiment_summary["test_accuracy_metrics_trt"] = metrics
        # experiment_summary["memory"] = {
        #     "engine_footprint": engine_footprint,
        #     "inference_memory_stats": memory_inference_stats
        # }
    print(f"Populated experiment summary: {experiment_summary}")

    import analyze_trt_csv_profile as analyze_trt_csv_profile
    
    print(f"Analyzing TensorRT CSV profile: {csv_filename}")
    logger.info(f"Analyzing TensorRT CSV profile: {csv_filename}")
    
    # Convert to nested format
    result = analyze_trt_csv_profile.convert_trt_csv_to_perfetto_json(csv_filename, f"{csv_filename[:-4]}_perfetto.json")
    analyze_trt_csv_profile.save_nested_profile(result["nested_profile"], f"{csv_filename[:-4]}_nested.json")
    
    top_layers, stage_times = analyze_trt_csv_profile.analyze_trt_csv_profile(csv_filename)
    # if tb_writer is not None:
    #     tb_writer.add_scalar("TensorRT Benchmark/Inference_Time", t_per_inference, global_step=0)
    #     tb_writer.add_scalar("TensorRT Benchmark/Throughput", throughput, global_step=0)
    #     tb_writer.add_scalar("TensorRT Benchmark/Avg_Latency", avg, global_step=0)
    #     tb_writer.add_scalar("TensorRT Benchmark/p95_Latency", p95, global_step=0)
    #     tb_writer.add_scalar("TensorRT Benchmark/IPS", ips, global_step=0)
    #     tb_writer.add_text("TensorRT benchmark/Top_Layers:", top_layers.to_string(), global_step=0)
    #     tb_writer.add_text("TensorRT benchmark/Stage_Times:", stage_times.to_string(), global_step=0)

    print(f"Top Layers:\n{top_layers}")
    logger.info(f"Top Layers:\n{top_layers}")
    print(f"Stage Times:\n{stage_times}")
    logger.info(f"Stage Times:\n{stage_times}")
    
    formatted_string = measure.format_dict_multiline(experiment_summary)
    print(formatted_string)

    print(f"Experiment summary: {formatted_string}")
    logger.info(f"Experiment summary: {formatted_string}")
    # save to file as json
    i = 0
    summary_file = os.path.join(args.output_dir, f"{args.slurm_job_id}_{args.experiment_id}_tensorRT_summary_{i:02d}.json")

    # if summary_file already exist, create new file with _01, _02, etc. suffix appended
    # if os.path.exists(summary_file):
        # base, ext = os.path.splitext(summary_file)
    while os.path.exists(summary_file):
        summary_file = f"{summary_file[:-8]}_{i:02d}.json"
        # summary_file = f"{base}_{i:02d}{ext}"
        i += 1

    with open(summary_file, 'w') as f:
        json.dump(experiment_summary, f, indent=4, default=str)

    WORLD_SIZE = os.environ.get('WORLD_SIZE', 1)
    print(f"Process {os.getpid()}: [Rank {local_rank}/{WORLD_SIZE}]: Onnx evaluation completed.")


    # import parse_training_log
    # summary = parse_training_log.do_log_parsing(
    #     filename=log_file, json_filename=summary_file)
    # logger.info(f"Parsed training log summary: {summary}")
    

if __name__ == "__main__":

    print(f"running command: ") 
    print(f"python ./OpenStereo_forked/tools/onnx_evaluation_tensorRT.py {sys.argv[1:]}")

    args, cfgs = config_parsing.parse_config()
    
    if args.onnx_file is not None:
        onnx_file = args.onnx_file
    else:
        onnx_file = os.path.join(args.output_dir, f"{args.extra_tag}.onnx")

    print(f"ONNX file: {onnx_file}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created new output directory: {args.output_dir}")


    # Run TensorRT benchmark
    run_trt_benchmark(onnx_file, args, cfgs)
