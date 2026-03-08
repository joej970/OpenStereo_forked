from datetime import datetime
import os
import sys
import json
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from tools/ to project root
sys.path.insert(0, project_root)

from stereo.utils import common_utils
from stereo.datasets import build_dataloader
from easydict import EasyDict
import measure

import config_parsing

def extract_exp_id_exp_name_slurm_job_id(filename):
    """
    Extracts exp_id, exp_name, and slurm_job_id from a filename of the form:
    <exp_id>_<exp_name>_<slurm_job_id>.<ext>

    Returns:
        exp_id (str), exp_name (str), slurm_job_id (str)
    """
    import os
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split('_')
    if len(parts) < 3:
        print("Filename does not contain expected exp_id, exp_name, and slurm_job_id parts.")
        return "unknown_id", "unknown_name", "0"
    exp_id = parts[0]
    slurm_job_id = parts[-1]
    exp_name = '_'.join(parts[1:-1])
    return exp_id, exp_name, slurm_job_id

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
    print(f"Logging to file: {log_file}")

    # log_file = os.path.join(args.output_dir, 'tensorRT_eval_{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), local_rank))
    logger = common_utils.create_logger(log_file, rank=local_rank)

    test_set, test_loader, test_sampler = build_test_loader(args, cfgs)
    logger.info('Total samples for test dataset: %d' % (len(test_set)))

    print(f"Running TensorRT benchmark on ONNX file: {onnx_file}")
    logger.info(f"Running TensorRT benchmark on ONNX file: {onnx_file}")
    # csv_filename = f"trt_benchmark_{args.slurm_job_id}.csv"
    csv_filename = os.path.join(args.output_dir, f"{args.experiment_id}_{args.slurm_job_id}_trt_benchmark.csv")

    # idx = 0
    experiment_summary = {}

    trt_bench = inference_benchmarking.trt_build_engine(onnx_file, csv_filename=csv_filename, fp16=True, data_loader=test_loader, cfgs=cfgs)

        # runs = args.onnx_eval_runs if hasattr(args, 'onnx_eval_runs') else 3
        # for run_idx in range(runs):
        # # while os.path.exists(onnx_file):
        #     avg, p95, ips, metrics, engine_footprint, memory_inference_stats = inference_benchmarking.trt_benchmark(trt_bench)

    runs = args.onnx_eval_runs if hasattr(args, 'onnx_eval_runs') else 3
    for run_idx in range(runs):
    # while os.path.exists(onnx_file):
        avg, p95, ips, metrics, engine_footprint, memory_inference_stats = inference_benchmarking.trt_benchmark(trt_bench)

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

    print(f"Top Layers:\n{top_layers.to_string()}")
    logger.info(f"Top Layers:\n{top_layers.to_string()}")
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

def seconds_to_hms(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def run_trt_benchmarks(onnx_files, args, cfgs):

    import inference_benchmarking
    import time

    time_start = time.time()

    experiment_summary = {}

    local_rank = 0

    log_file = os.path.join(args.output_dir, f"{args.slurm_job_id}_{args.experiment_id}.log")
    print(f"Logging to file: {log_file}")

    # log_file = os.path.join(args.output_dir, 'tensorRT_eval_{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), local_rank))
    logger = common_utils.create_logger(log_file, rank=local_rank)

    test_set, test_loader, test_sampler = build_test_loader(args, cfgs)
    logger.info('Total samples for test dataset: %d' % (len(test_set)))

    one_line_results = []


    runs = args.onnx_eval_runs if hasattr(args, 'onnx_eval_runs') else 3
    for o, onnx_file in enumerate(onnx_files):

        if not os.path.exists(onnx_file):
            print(f"ONNX file does not exist: {onnx_file}, skipping...")
            logger.info(f"ONNX file does not exist: {onnx_file}, skipping...")
            continue

        exp_id, exp_name, slurm_job_id = extract_exp_id_exp_name_slurm_job_id(onnx_file)


        print(f"Running TensorRT benchmark on ONNX file: {onnx_file}")
        logger.info(f"Running TensorRT benchmark on ONNX file: {onnx_file}")
        logger.info(f"exp_id: {exp_id}, exp_name: {exp_name}, slurm_job_id: {slurm_job_id}")
        csv_filename = os.path.join(args.output_dir, f"{exp_id}_{exp_name}_{slurm_job_id}_trt_benchmark.csv")

        experiment_summary = {}

        ips_arr = []
        avg_lat_arr = []
        p95_lat_arr = []
        engine_footprint_arr = []
        memory_inference_stats_arr = []

        logger.info(f"Building TensorRT engine for ONNX file.")

        trt_bench = inference_benchmarking.trt_build_engine(onnx_file, csv_filename=csv_filename, fp16=True, data_loader=test_loader, cfgs=cfgs)
        logger.info(f"Completed building TensorRT engine.")
        
        for run_idx in range(runs):
        # while os.path.exists(onnx_file):
            avg, p95, ips, metrics, engine_footprint, memory_inference_stats = inference_benchmarking.trt_benchmark(trt_bench)

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

            ips_arr.append(ips)
            avg_lat_arr.append(avg)
            p95_lat_arr.append(p95)
            engine_footprint_arr.append(engine_footprint["total_model_memory_mb"])
            memory_inference_stats_arr.append(memory_inference_stats["mem_mb_mean"])

            # idx += 1
            # onnx_file = f"{onnx_file[:-8]}_run{idx}.onnx"

            print(f"Completed TensorRT benchmark run {run_idx+1}/{runs}: summary: {summary}")
            logger.info(f"Completed TensorRT benchmark run {run_idx+1}/{runs}: summary: {summary}")

        trt_bench.cleanup()
        del trt_bench
        import gc
        gc.collect()

        ips_arr = np.array(ips_arr)
        avg_lat_arr = np.array(avg_lat_arr)
        p95_lat_arr = np.array(p95_lat_arr)
        engine_footprint_arr = np.array(engine_footprint_arr)
        memory_inference_stats_arr = np.array(memory_inference_stats_arr)


        # print(f"ips_arr.type: {type(ips_arr)}, ips_arr: {ips_arr}")
        # print(f"avg_lat_arr.type: {type(avg_lat_arr)}, avg_lat_arr: {avg_lat_arr}")
        # print(f"p95_lat_arr.type: {type(p95_lat_arr)}, p95_lat_arr: {p95_lat_arr}")
        # print(f"engine_footprint_arr.type: {type(engine_footprint_arr)}, engine_footprint_arr: {engine_footprint_arr}")
        # print(f"np.array(engine_footprint_arr).type: {type(engine_footprint_arr)}, np.array(engine_footprint_arr): {engine_footprint_arr}")
        # print(f"memory_inference_stats_arr.type: {type(memory_inference_stats_arr)}, memory_inference_stats_arr: {memory_inference_stats_arr}")
        # print(f"np.array(memory_inference_stats_arr).type: {type(memory_inference_stats_arr)}, np.array(memory_inference_stats_arr): {memory_inference_stats_arr}")


        # one_line_result = f"{exp_id}, {slurm_job_id}, {', '.join(f'{x:.2f}' for x in np.array(ips_arr))}, {np.mean(ips_arr):.2f}, {np.std(ips_arr):.2f}, , , , , {np.min(np.array(engine_footprint_arr)):.2f}, {np.min(np.array(memory_inference_stats_arr)):.2f}"

        one_line_result = f"{exp_id}, {slurm_job_id}, "
        one_line_result += f"{', '.join(f'{x:.2f}' for x in np.array(ips_arr))}"
        one_line_result += f", {np.mean(ips_arr):.2f}, {np.std(ips_arr):.2f}, , , , , "
        one_line_result += f"{np.min(np.array(engine_footprint_arr)):.2f}, "
        one_line_result += f"{np.min(np.array(memory_inference_stats_arr)):.2f}"

        print(f"Experiment {o+1}/{len(onnx_files)} result summary:\n{one_line_result}")
        logger.info(f"Experiment {o+1}/{len(onnx_files)} result summary:\n{one_line_result}")

        one_line_results.append(one_line_result)

        print(f"Populated experiment summary: {experiment_summary}")
        logger.info(f"Populated experiment summary: {experiment_summary}")

        import analyze_trt_csv_profile as analyze_trt_csv_profile
        
        print(f"Analyzing TensorRT CSV profile: {csv_filename}")
        logger.info(f"Analyzing TensorRT CSV profile: {csv_filename}")
        
        # Convert to nested format
        result = analyze_trt_csv_profile.convert_trt_csv_to_perfetto_json(csv_filename, f"{csv_filename[:-4]}_perfetto.json")
        analyze_trt_csv_profile.save_nested_profile(result["nested_profile"], f"{csv_filename[:-4]}_nested.json")
        
        top_layers, stage_times = analyze_trt_csv_profile.analyze_trt_csv_profile(csv_filename)

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

        # WORLD_SIZE = os.environ.get('WORLD_SIZE', 1)
        # print(f"Process {os.getpid()}: [Rank {local_rank}/{WORLD_SIZE}]: Onnx evaluation completed.")

    seconds_spent = time.time() - time_start
    time_spent = seconds_to_hms(seconds_spent)

    print(f"All done. Total time for all ({len(onnx_files)}x{runs}={len(onnx_files)*runs} runs) TensorRT benchmarks: {time_spent}")

    if args.node_id is not None:
        print(f"Node ID: {args.node_id}")
        logger.info(f"Node ID: {args.node_id}")

    column_names = "exp_id, slurm_job_id, "
    column_names += ", ".join([f"run_{i:02d}_ips" for i in range(runs)])
    column_names += ", mean_ips, std_ips, , , , , min_engine_footprint_mb, min_inference_memory_mb"

    print(f"All experiments completed. Summary of results:\n{column_names}")
    logger.info(f"All experiments completed. Summary of results:\n{column_names}")
    for line in one_line_results:
        print(line)
        logger.info(line)
    

if __name__ == "__main__":

    print(f"running command: ") 
    print(f"python ./OpenStereo_forked/tools/onnx_evaluation_tensorRT.py {sys.argv[1:]}")

    args, cfgs = config_parsing.parse_config()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created new output directory: {args.output_dir}")
    
    # Running TensorRT benchmarks on multiple ONNX files
    if args.trt_onnx_list is not None:
        print(f"Running TensorRT benchmarks on ONNX file list: {args.trt_onnx_list}")
        run_trt_benchmarks(args.trt_onnx_list, args, cfgs)
        sys.exit(0)

    # Running TensorRT benchmark on a single ONNX file
    if args.onnx_file is not None:
        onnx_file = args.onnx_file
    # Running TensorRT benchmark on a single ONNX file specified yaml config
    else:
        onnx_file = os.path.join(args.output_dir, f"{args.extra_tag}.onnx")

    print(f"ONNX file: {onnx_file}")

    # Run TensorRT benchmark
    run_trt_benchmark(onnx_file, args, cfgs)
