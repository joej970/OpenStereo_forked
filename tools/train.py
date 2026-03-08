# @Time    : 2023/8/28 22:18
# @Author  : zhangchenming
import sys
import os
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

import config_parsing



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
    args, cfgs = config_parsing.parse_config()

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
    # args.output_dir = str(os.path.join(args.save_root_dir, args.exp_group_path, args.tag, args.extra_tag))
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
    log_file = os.path.join(args.output_dir, f"{args.slurm_job_id}_{args.experiment_id}.log")
    # log_file = os.path.join(args.output_dir, 'train_{}.log'.format( group_rank))
    # log_file = os.path.join(args.output_dir, 'train_{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), group_rank))
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
    model_trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer) 
    # model_trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer, enable_profiler=args.enable_profiler) ,

    print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Trainer built successfully.")

    tbar = tqdm.trange(model_trainer.last_epoch + 1, model_trainer.total_epochs,
                       desc='epochs', dynamic_ncols=True, disable=(local_rank != 0),
                       bar_format='{l_bar}{bar}{r_bar}\n')
    # train loop
    best_epoch = {'idx': 0, 'epe': 1e6}
    
    # Start timing the training
    training_start_time = datetime.datetime.now()
    print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Starting training loop at {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    training_epoch_count = 0
    evaluation_epoch_count = 0

    # perform
    print("Starting evaluation with epoch=-1 to get measurements before GPU memory gets fragmented by training.")
    model_trainer.evaluate(-1)
    print("Initial evaluation completed. Starting training epochs.")
    
    train_losses = []
    train_lrs = []
    train_epochs = []
    eval_epes = []
    eval_epochs = []

    current_epoch = 0

    for current_epoch in tbar:
        model_trainer.train(current_epoch, tbar)
        training_epoch_count += 1
        model_trainer.save_ckpt(current_epoch)

        train_losses.append(model_trainer.train_losses.get(current_epoch, None))
        train_lrs.append(model_trainer.train_lrs.get(current_epoch, None))
        train_epochs.append(current_epoch)



        if current_epoch % cfgs.TRAINER.EVAL_INTERVAL == 0 or current_epoch == model_trainer.total_epochs - 1:
            model_trainer.evaluate(current_epoch)
            evaluation_epoch_count += 1
            current_epe = model_trainer.eval_epes.get(current_epoch, None)
            if current_epe is None:
                print(f"Warning: EPE for epoch {current_epoch} not found")
                print(f"Found only: {model_trainer.eval_epes}")
            else:
                eval_epes.append(current_epe)
                eval_epochs.append(current_epoch)

                common_utils.draw_eval_epe(args.output_dir, args.slurm_job_id, args.experiment_id, eval_epochs=eval_epochs, eval_epes=eval_epes)

                if current_epe < best_epoch['epe']:
                    best_epoch = {'idx': current_epoch, 'epe': current_epe}
                    model_trainer.save_best_pth(current_epoch)
                    print(f"Saving best model for epoch {best_epoch} with EPE {current_epe}")

        common_utils.draw_train_loss_lr(
            args.output_dir, args.slurm_job_id, args.experiment_id,
            train_epochs = train_epochs,
            train_losses = train_losses,
            train_lrs = train_lrs,
            eval_epochs = eval_epochs,
            eval_epes = eval_epes
        )

    # End timing the training
    training_end_time = datetime.datetime.now()
    training_duration = training_end_time - training_start_time
    total_seconds = training_duration.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    total_epochs = training_epoch_count + evaluation_epoch_count
    avg_time_per_training_epoch = total_seconds / training_epoch_count if training_epoch_count > 0 else 0
    
    training_summary = (
        f"\n{'='*80}\n"
        f"TRAINING COMPLETED\n"
        f"{'='*80}\n"
        f"Training started:  {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Training ended:    {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_seconds:.1f} seconds)\n"
        f"Training epochs: {training_epoch_count}, Evaluation epochs: {evaluation_epoch_count}\n"
        f"{training_epoch_count}+{evaluation_epoch_count}={total_epochs} total epochs\n"
        f"Average time per training+evaluation epoch: {avg_time_per_training_epoch:.1f} seconds\n"
        f"{'='*80}"
    )
    
    print(training_summary)
    logger.info(training_summary)

    print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Training loop completed. Best epoch {best_epoch} with epe {best_epoch['epe']:.4f}")

    evaluation_message = f"Model evaluation results: {model_trainer.eval_epes}"
    print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: {evaluation_message}")
    logger.info(evaluation_message)

    message_idx = f"Final best epoch idx: {best_epoch['idx']}"
    message_epe = f"Final best epoch epe: {best_epoch['epe']:.4f}"
    #2025-08-28 10:46:01,444   INFO  Final best epoch idx: 45
    #2025-08-28 10:46:01,444   INFO  Final best epoch epe: 4.356
    logger.info(message_idx)
    logger.info(message_epe)

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

    
    # # accuracy (EPE and other benchmarking)
    # print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Starting testing phase.")
    # test_results = model_trainer.test(current_epoch)

    # experiment_summary["test_accuracy_metrics"] = test_results

    del model_trainer
    # inference time benchmark

    args.run_mode = 'eval'
    infer_model_trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer) 
    model = infer_model_trainer.model

    best_pth_name = os.path.join(args.ckpt_dir, 'best_model.pth')
    try:
        # best_model_filename = 'best_model.pth'
        logger.info('Loading best model from checkpoint %s' % best_pth_name)
        if not os.path.isfile(best_pth_name):
            raise FileNotFoundError
        common_utils.load_params_from_file(
            model, best_pth_name, device='cuda:%d' % local_rank,
            dist_mode=args.dist_mode, logger=logger, strict=False)
        print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Successfully loaded best model from checkpoint {best_pth_name}.")
    except Exception as e:
        print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Failed to load best model from checkpoint {best_pth_name}. Continuing with the last epoch model.")
        logger.info(e)
        logger.info(f"Failed to load best model from checkpoint {best_pth_name}. Continuing with the last epoch model.")


    # accuracy (EPE and other benchmarking)
    print(f"Process {os.getpid()}: [Rank {global_rank}/{WORLD_SIZE}]: Starting testing phase.")
    test_results = infer_model_trainer.test(current_epoch)

    experiment_summary["test_accuracy_metrics"] = test_results

    # here you should read vest model from ptx



    
    if args.dist_mode:
        model = model.module  # Get the underlying model if using DDP

    # Convert model output to just the tensor (avoid dict)
    class WrappedModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.additional_depth_src = getattr(base_model, 'additional_depth_src', False)
            print(f"the base model has source: {self.additional_depth_src}")

        def forward(self, left, right, depth_source = None):
            if self.additional_depth_src:
                if(depth_source is None):
                    raise ValueError("depth_source must be provided when additional_depth_src is enabled")
                data = {"left": left, "right": right, "depth_src_0": depth_source}
            else:
                data = {"left": left, "right": right}
            return self.base_model(data)["disp_pred"]  # single output
    
    model = WrappedModel(model).cuda().eval()

    def set_eval_recursive(model):
        for module in model.modules():
            module.eval()

    set_eval_recursive(model)

    # model.additional_depth_src = additional_depth_src

    # additional_depth_src = False
    print("")
    print("1:")
    if hasattr(model, 'additional_depth_src'):
        print("Model has additional depth source")
        if model.additional_depth_src:
            # additional_depth_src = True
            print("Additional depth source is enabled")
        else:
            # additional_depth_src = False
            print("Additional depth source is disabled")
    else:
        print("Model does not have additional depth source")
    print("")

    test_resolution = cfgs.DATA_CONFIG.DATA_TRANSFORM.TESTING[0].SIZE
    # shape = [1, 3, 544, 960]  # keep batchsize 1
    shape = [1, 3, test_resolution[0], test_resolution[1]]  # keep batchsize 1
    print(f"Test Resolution is {shape}.") 
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

    if args.onnx_file is not None:
        onnx_file = args.onnx_file
    else:
        onnx_file = os.path.join(args.output_dir, f"{args.extra_tag}.onnx")

    with torch.no_grad():
        measure.export_to_onnx(model = model, input_shape = shape, onnx_path = onnx_file, use_fp16=False, dynamic_batch = False, opset_version=17)


    if tb_writer is not None:
        tb_writer.close()



# python -m debugpy --listen 0.0.0.0:5678 --wait-for-client ./tools/train.py

if __name__ == '__main__':
    # import sys
    # # Manually supply arguments for debugging
    # sys.argv = [
    #     'python tools/train.py',  # Script name
    #     # '--cfg_file',      './cfgs/onestereo/one_stereo_s_sceneflow_dev.yaml', 
    #     # '--cfg_file',      './cfgs/onestereo/one_stereo_local_mobileone.yaml', 
    #     '--cfg_file',      './cfgs/LeanStereo/300_LeanStereo_sceneflow.yaml', 
    #     # '--data_cfg_file', './cfgs/onestereo/one_stereo_s_sceneflow_hpc_config.yaml', 
    #     # '--data_cfg_file', './cfgs/onestereo/dataset_sceneflow_dev.yaml', 
    #     '--data_cfg_file', './data/SceneFlow/sceneflow_hpc_finalpass_dev.yaml', 
    #     '--workers', '2',
    #     '--force_override'
    # ]
    print("Running training script with the following arguments:")
    for arg in sys.argv[1:]:
        print(arg)
    
    print("Current working directory:", os.getcwd())
    print("Python executable:", sys.executable)
    main()
