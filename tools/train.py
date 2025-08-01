# @Time    : 2023/8/28 22:18
# @Author  : zhangchenming
import sys
import os
import argparse
import datetime
import tqdm
from easydict import EasyDict
import socket
import measure

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

# sys.path.insert(0, './')
# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from tools/ to project root
sys.path.insert(0, project_root)
print(f"Added to Python path: {project_root}")
print(f"Current working directory: {os.getcwd()}")

from stereo.utils import common_utils
from stereo.modeling import build_trainer
from cfgs.data_basic import DATA_PATH_DICT

def recursive_merge(base, update):
    """
    Recursively merge two dictionaries. The `update` dictionary will overwrite or add to the `base` dictionary.
    """
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            # If both base and update have a dictionary for this key, merge them recursively
            recursive_merge(base[key], value)
        else:
            # Otherwise, overwrite or add the value
            base[key] = value

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # mode
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify the config for training')
    parser.add_argument('--data_cfg_file', type=str, default=None, required=True, help='specify the dataset config for training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    # save path
    parser.add_argument('--save_root_dir', type=str, default='./output', help='save root dir for this experiment')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    # dataloader
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='data loader pin memory')
    parser.add_argument('--force_override', action='store_true', default=False, help='Force overwrite the existing experiment')
    parser.add_argument('--backend', type=str, default='nccl', help='gpu intercommunication backend, default is nccl, options: gloo, nccl')
    parser.add_argument('--enable_profiler', action='store_true', help='Enable torch.profiler for debugging')
    # batch_size argument
    parser.add_argument('--batch_size', type=int, default=None, help='Override BATCH_SIZE_PER_GPU in config')
    parser.add_argument('--overide_epoch', type=int, default=None, help='Override EPOCH in config.')
    parser.add_argument('--slurm_job_id', type=str, default=None, help='Optional: SLURM job ID for logging')
    parser.add_argument('--experiment_id', type=str, default=None, help='Optional: Experiment ID for logging')


    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

     # Load the data config file
    if args.data_cfg_file:
        data_yaml_config = common_utils.config_loader(args.data_cfg_file)
        data_cfgs = EasyDict(data_yaml_config)

        # # Replace or merge only the DATA_INFOS section
        # if 'DATA_CONFIG' in data_cfgs and 'DATA_INFOS' in data_cfgs.DATA_CONFIG:
        #     cfgs.DATA_CONFIG.DATA_INFOS = data_cfgs.DATA_CONFIG.DATA_INFOS  # Replace DATA_INFOS
        # if 'OPTIMIZATION' in data_cfgs:
        #     cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU = data_cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU

        # Recursively merge data_cfgs into cfgs
        recursive_merge(cfgs, data_cfgs)

        # --- Override BATCH_SIZE_PER_GPU if --batch_size is provided ---
    if args.batch_size is not None:
        if 'OPTIMIZATION' in cfgs and hasattr(cfgs.OPTIMIZATION, 'BATCH_SIZE_PER_GPU'):
            cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU = args.batch_size
        elif 'OPTIMIZATION' in cfgs:
            cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU = args.batch_size
        else:
            cfgs.OPTIMIZATION = EasyDict({'BATCH_SIZE_PER_GPU': args.batch_size})

    if args.overide_epoch is not None:
        if 'OPTIMIZATION' in cfgs and hasattr(cfgs.OPTIMIZATION, 'NUM_EPOCHS'):
            cfgs.OPTIMIZATION.NUM_EPOCHS = args.overide_epoch
        elif 'OPTIMIZATION' in cfgs:
            cfgs.OPTIMIZATION.NUM_EPOCHS = args.overide_epoch
        else:
            cfgs.OPTIMIZATION = EasyDict({'NUM_EPOCHS': args.overide_epoch})


    dataset_names = [x.DATASET for x in cfgs.DATA_CONFIG.DATA_INFOS]
    unique_dataset_names = list(set(dataset_names))
    if len(unique_dataset_names) == 1:
        exp_dataset_dir = unique_dataset_names[0]
    else:
        exp_dataset_dir = 'MultiDataset'
    args.exp_group_path = os.path.join(exp_dataset_dir, cfgs.MODEL.NAME)
    args.tag = os.path.basename(args.cfg_file)[:-5]
    message = f"Experiment group path: {args.exp_group_path}, tag: {args.tag}"
    print(message)

    for each in cfgs.DATA_CONFIG.DATA_INFOS:
        dataset_name = each.DATASET
        if dataset_name == 'KittiDataset':
            dataset_name = 'KittiDataset15' if 'kitti15' in each.DATA_SPLIT.EVALUATING else 'KittiDataset12'
        # each.DATA_PATH = DATA_PATH_DICT[dataset_name]

    args.run_mode = 'train'
    # print(f"data_path: {cfgs.DATA_CONFIG.DATA_INFOS[0].DATA_PATH}")
   

    return args, cfgs

# slurm environment variables
# WORLD_SIZE = int(os.environ['SLURM_NTASKS'])
# WORLD_RANK = int(os.environ['SLURM_PROCID'])
# LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
# torchrun environment variables
# WORLD_SIZE = int(os.environ['WORLD_SIZE'])
# WORLD_RANK = int(os.environ['RANK'])
# LOCAL_RANK = int(os.environ['LOCAL_RANK'])


# print("tasks per node: ", os.environ['SLURM_TASKS_PER_NODE'])
# GROUP_RANK = WORLD_RANK/2


def log_configs_to_tensorboard(cfgs, tb_writer, pre='cfgs', step=0):
    """Recursively log config parameters to tensorboard as text."""
    for key, val in cfgs.items():
        if isinstance(val, EasyDict):
            log_configs_to_tensorboard(val, tb_writer, pre=pre + '.' + key, step=step)
            continue
        tb_writer.add_text(f"Config/{pre}.{key}", str(val), global_step=step)

def main():
    print("In main() function of train.py")
    args, cfgs = parse_config()
    if args.dist_mode:
        id = os.getpid()
        print(f"Process {id}: Starting distributed process...")
        # WORLD_SIZE = int(os.environ['SLURM_NTASKS'])
        # WORLD_RANK = int(os.environ['SLURM_PROCID'])
        # LOCAL_RANK = int(os.environ['SLURM_LOCALID'])


        print(f"Process {id}: env vars: LOCAL_RANK: {int(os.environ['RANK'])}, WORLD_RANK: {int(os.environ['RANK'])}, WORLD_SIZE: {int(os.environ['WORLD_SIZE'])}")

        print(f"Process {id}: prior to init_process_group")
        # dist.init_process_group(backend=args.backend,init_method='env://', world_size=WORLD_SIZE, rank=WORLD_RANK)
        dist.init_process_group(backend=args.backend)
        # dist.init_process_group(backend='nccl',init_method='env://', world_size=WORLD_SIZE, rank=WORLD_RANK)


        print(f"Process {id}: after init_process_group")

        WORLD_RANK = dist.get_rank()
        LOCAL_RANK = WORLD_RANK
        WORLD_SIZE = dist.get_world_size()

        if LOCAL_RANK != int(os.environ.get("LOCAL_RANK", 0)):
            raise ValueError(f"Local rank mismatch: expected {os.environ.get('LOCAL_RANK', 0)}, got {LOCAL_RANK}")
        
        local_rank = LOCAL_RANK
        global_rank = WORLD_RANK
        group_rank = int(global_rank//2)

        hostname = socket.gethostname()
        
        print(f"[Rank {LOCAL_RANK}/{WORLD_SIZE}] torch sees {torch.cuda.device_count()} GPUs")

        print(f"Process {id}: [Rank {global_rank}/{WORLD_SIZE}({WORLD_SIZE})] Hello from {hostname}. Env vars: Local Rank: {local_rank}, Global Rank: {global_rank}, group rank: {group_rank}, world_size: {WORLD_RANK}, dist.rank: {dist.get_rank()}, dist.world_size: {dist.get_world_size()}")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    else:
        local_rank = 0
        global_rank = 0
        group_rank = 0
        WORLD_SIZE = 1

    # env
    torch.cuda.set_device(local_rank)
    if args.fix_random_seed:
        seed = 0 if not args.dist_mode else dist.get_rank()
        common_utils.set_random_seed(seed=seed)

    # savedir
    args.output_dir = str(os.path.join(args.save_root_dir, args.exp_group_path, args.tag, args.extra_tag))
    if global_rank == 0:
        if os.path.exists(args.output_dir) and args.extra_tag != 'debug' and cfgs.MODEL.CKPT == -1:
            if args.force_override:
                print(f"Force override the existing experiment: {args.output_dir}")
                import shutil
                shutil.rmtree(args.output_dir)
                os.makedirs(args.output_dir, exist_ok=True)
            else:
                raise Exception(f"There is already an exp with this name: {args.output_dir}")
    if args.dist_mode:
        print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}] Waiting for all ranks to reach the barrier.")
        dist.barrier()
        if global_rank == 0:
            print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: ✅ All ranks reached the barrier. DDP is working!")
        else:
            print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Continue after 1st barrier.")

    args.ckpt_dir = os.path.join(args.output_dir, 'ckpt')
    if not os.path.exists(args.ckpt_dir) and local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    if global_rank == 0:
        common_utils.backup_source_code(os.path.join(args.output_dir, 'code'))
    if args.dist_mode:
        dist.barrier()
        print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Continue after 2nd barrier.")

    # tensorboard
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard')) if global_rank == 0 else None
    if tb_writer is not None:
        # Log hyperparameters
        tb_writer.add_hparams(
            hparam_dict=vars(args),
            metric_dict={'hparam/epoch': 0, 'hparam/loss': 0},
            run_name=f"{args.tag}_{args.extra_tag}"
        )
        # Log args as text
        args_text = "\n".join([f"{key}: {val}" for key, val in vars(args).items()])
        tb_writer.add_text("Arguments", args_text, global_step=0)
        if args.slurm_job_id is not None:
            tb_writer.add_text("System/SLURM_Job_ID", str(args.slurm_job_id), global_step=0)
        # Log all cfgs parameters recursively to tensorboard
        log_configs_to_tensorboard(cfgs, tb_writer, pre='cfgs', step=0)
    # logger
    log_file = os.path.join(args.output_dir, 'train_{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), group_rank))
    logger = common_utils.create_logger(log_file, rank=local_rank)
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    if args.slurm_job_id is not None:
        logger.info(f"SLURM Job ID: {args.slurm_job_id}")
    common_utils.log_configs(cfgs, logger=logger)

     # --- Log hostname and GPU info ---
    hostname = socket.gethostname()
    num_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)] if num_gpus > 0 else ["No CUDA devices"]
    num_cpus = os.cpu_count()

    logger.info(f"Hostname: {hostname}")
    logger.info(f"Number of CUDA devices: {num_gpus}")
    logger.info(f"CUDA device(s): {', '.join(gpu_names)}")
    logger.info(f"Number of CPU processors: {num_cpus}")

    if tb_writer is not None:
        tb_writer.add_text("System/Hostname", hostname, global_step=0)
        tb_writer.add_text("System/Num_GPUs", str(num_gpus), global_step=0)
        tb_writer.add_text("System/GPU_Names", "<br>".join(gpu_names), global_step=0)
        tb_writer.add_text("System/Num_CPUs", str(num_cpus), global_step=0)
    # --- End log hostname and GPU info ---

    if global_rank == 0:
        os.system('cp %s %s' % (args.cfg_file, args.output_dir))

    print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Building trainer.")
    # trainer
    model_trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer, enable_profiler=args.enable_profiler) 

    print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Trainer built successfully.")

    tbar = tqdm.trange(model_trainer.last_epoch + 1, model_trainer.total_epochs,
                       desc='epochs', dynamic_ncols=True, disable=(local_rank != 0),
                       bar_format='{l_bar}{bar}{r_bar}\n')
    # train loop
    print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Starting training loop.")
    for current_epoch in tbar:
        model_trainer.train(current_epoch, tbar)
        model_trainer.save_ckpt(current_epoch)
        if current_epoch % cfgs.TRAINER.EVAL_INTERVAL == 0 or current_epoch == model_trainer.total_epochs - 1:
            model_trainer.evaluate(current_epoch)

    print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Training loop completed.")

    experiment_summary = {
        "experiment_ID": args.experiment_id,
        "valid_accuracy_metrics": {},
        "test_accuracy_metrics": {},
        "test_accuracy_metrics_trt": {},
        "parameter_count": {},
        "inference_benchmark": {
            "shape": None,  # Will be filled later
            "pyTorch_inference": {},
            "TorchScript_inference": {},
            "Onnx_trt_inference": {}
        }
    }

    # accuracy (EPE and other benchmarking)
    print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Starting testing phase.")
    test_results = model_trainer.test(current_epoch)

    experiment_summary["test_accuracy_metrics"] = test_results
    
    # Convert model output to just the tensor (avoid dict)
    class WrappedModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, left, right):
            data = {"left": left, "right": right}
            return self.base_model(data)["disp_pred"]  # single output
    
    # inference time benchmark
    model = model_trainer.model

    if args.dist_mode:
        model = model.module  # Get the underlying model if using DDP

    model = WrappedModel(model).cuda().eval()

    def set_eval_recursive(model):
        for module in model.modules():
            module.eval()

    set_eval_recursive(model)


    shape = [1, 3, 544, 960]  # keep batchsize 1
    message = f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Benchmarking model with shape {shape}"
    logger.info(message)
    experiment_summary["inference_benchmark"]["shape"] = shape

    print(f"Measuring pyTorch inference time")
    logger.info(f"Measuring pyTorch inference time")
    message_1, message_2, flops, params = measure.measure(model, shape)
    print(message_1)
    logger.info(message_1)
    print(message_2)
    logger.info(message_2)
    experiment_summary["parameter_count"]["num_params"] = params
    experiment_summary["parameter_count"]["flops"] = flops

    message_1, message_2, t_per_inference, throughput = measure.infer_time(model, shape) # keep batchsize 1
    logger.info(message_1)
    logger.info(message_2)
    print(message_1)
    print(message_2)
    experiment_summary["inference_benchmark"]["pyTorch_inference"] = {
        "time_per_inference_ms": t_per_inference,
        "throughput": throughput
    }

    message = f"Now with TorchScript model"
    logger.info(message)
    print(message)
    message_3, throughput = measure.infer_time_torch_script(model, shape) # keep batchsize 1
    logger.info(message_3)
    print(message_3)
    experiment_summary["inference_benchmark"]["TorchScript_inference"] = {
        "time_per_inference_ms": 1000/throughput,
        "throughput": throughput
    }

    onnx_file = os.path.join(args.output_dir, f"{args.extra_tag}.onnx")
    measure.export_to_onnx(model = model, input_shape = shape, onnx_path = onnx_file, use_fp16=False, dynamic_batch = False, opset_version=11)

    from inference_benchmarking import trt_benchmark
    
    print(f"Running TensorRT benchmark on ONNX file: {onnx_file}")
    logger.info(f"Running TensorRT benchmark on ONNX file: {onnx_file}")
    # csv_filename = f"trt_benchmark_{args.slurm_job_id}.csv"
    csv_filename = os.path.join(args.output_dir, f"trt_benchmark.csv")
    avg, p95, ips, metrics = trt_benchmark(onnx_file, csv_filename=csv_filename, fp16=True, data_loader=model_trainer.test_loader, cfgs=cfgs)


    experiment_summary["inference_benchmark"]["Onnx_trt_inference"] = {
        "avg_latency": avg,
        "p95_latency": p95,
        "inference": ips
    }
    experiment_summary["test_accuracy_metrics_trt"] = metrics


    import analyze_trt_csv_profile as analyze_trt_csv_profile
    
    print(f"Analyzing TensorRT CSV profile: {csv_filename}")
    logger.info(f"Analyzing TensorRT CSV profile: {csv_filename}")
    
    # Convert to nested format
    result = analyze_trt_csv_profile.convert_trt_csv_to_perfetto_json(csv_filename, f"{csv_filename[:-4]}_perfetto.json")
    analyze_trt_csv_profile.save_nested_profile(result["nested_profile"], f"{csv_filename[:-4]}_nested.json")
    
    top_layers, stage_times = analyze_trt_csv_profile.analyze_trt_csv_profile(csv_filename)
    if tb_writer is not None:
        tb_writer.add_scalar("TensorRT Benchmark/Inference_Time", t_per_inference, global_step=0)
        tb_writer.add_scalar("TensorRT Benchmark/Throughput", throughput, global_step=0)
        tb_writer.add_scalar("TensorRT Benchmark/Avg_Latency", avg, global_step=0)
        tb_writer.add_scalar("TensorRT Benchmark/p95_Latency", p95, global_step=0)
        tb_writer.add_scalar("TensorRT Benchmark/IPS", ips, global_step=0)
        tb_writer.add_text("TensorRT benchmark/Top_Layers:", top_layers.to_string(), global_step=0)
        tb_writer.add_text("TensorRT benchmark/Stage_Times:", stage_times.to_string(), global_step=0)


    
    formatted_string = measure.format_dict_multiline(experiment_summary)
    print(formatted_string)

    print(f"Experiment summary: {formatted_string}")
    logger.info(f"Experiment summary: {formatted_string}")


    print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Training completed. Exiting.")

    if tb_writer is not None:
        tb_writer.close()




# python -m debugpy --listen 0.0.0.0:5678 --wait-for-client ./tools/train.py

if __name__ == '__main__':
    # import sys
    # # Manually supply arguments for debugging
    # sys.argv = [
    #     'python tools/train.py',  # Script name
    #     # '--cfg_file',      './cfgs/onestereo/one_stereo_s_sceneflow_dev.yaml', 
    #     '--cfg_file',      './cfgs/onestereo/one_stereo_local_mobileone.yaml', 
    #     # '--data_cfg_file', './cfgs/onestereo/one_stereo_s_sceneflow_hpc_config.yaml', 
    #     '--data_cfg_file', './cfgs/onestereo/dataset_sceneflow_dev.yaml', 
    #     '--workers', '2',
    #     '--force_override'
    # ]
    print("Running training script with the following arguments:")
    for arg in sys.argv[1:]:
        print(arg)
    
    print("Current working directory:", os.getcwd())
    print("Python executable:", sys.executable)
    main()
