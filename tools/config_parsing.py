
import argparse
import os
import yaml
# from stereo.utils import common_utils

from easydict import EasyDict

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

def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    return src_cfgs

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
    # parser.add_argument('--enable_profiler', action='store_true', help='Enable torch.profiler for debugging')
    # batch_size argument
    parser.add_argument('--batch_size', type=int, default=None, help='Override BATCH_SIZE_PER_GPU in config')
    parser.add_argument('--onnx_eval_runs', type=int, default=3, help='Specifies number for ONNX runs for evaluation (default: 3)')
    parser.add_argument('--overide_epoch', type=int, default=None, help='Override EPOCH in config.')
    parser.add_argument('--override_epoch', type=int, default=None, help='Override EPOCH in config.')
    parser.add_argument('--slurm_job_id', type=str, default=None, help='Optional: SLURM job ID for logging')
    parser.add_argument('--experiment_id', type=str, default=None, help='Optional: Experiment ID for logging')
    parser.add_argument('--onnx_file', type=str, default=None, help='Optional: ONNX file name for evaluation')
    parser.add_argument('--lr', type=float, default=None, help='Optional: Specify learning rate to override the config learning rate')
    parser.add_argument('--amp', type=str, default=None, help='Optional: true/false to override AMP setting in config')
    # parser.add_argument('--')

    args = parser.parse_args()
    yaml_config = config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    if args.overide_epoch is not None: # back compatibility for typo
        args.override_epoch = args.overide_epoch

     # Load the data config file
    if args.data_cfg_file:
        data_yaml_config = config_loader(args.data_cfg_file)
        data_cfgs = EasyDict(data_yaml_config)

        # # Replace or merge only the DATA_INFOS section
        # if 'DATA_CONFIG' in data_cfgs and 'DATA_INFOS' in data_cfgs.DATA_CONFIG:
        #     cfgs.DATA_CONFIG.DATA_INFOS = data_cfgs.DATA_CONFIG.DATA_INFOS  # Replace DATA_INFOS
        # if 'OPTIMIZATION' in data_cfgs:
        #     cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU = data_cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU

        # Recursively merge data_cfgs into cfgs
        recursive_merge(cfgs, data_cfgs)

        # --- Override BATCH_SIZE_PER_GPU if --batch_size is provided ---

    if args.amp is not None:
        print(f"Overriding AMP setting in config with command-line argument: {args.amp}")
        if args.amp.lower() == 'true':
            cfgs.OPTIMIZATION.AMP = True
        elif args.amp.lower() == 'false':
            cfgs.OPTIMIZATION.AMP = False
        else:
            raise ValueError(f"Invalid value for --amp: {args.amp}. Use 'true' or 'false'.")

    if args.lr is not None:
        print(f"Overriding learning rate in config with command-line argument: {args.lr}")
        if 'OPTIMIZATION' in cfgs and hasattr(cfgs.OPTIMIZATION, 'LR'):
            print("Found existing LR in OPTIMIZATION, overriding it.")
            cfgs.OPTIMIZATION.OPTIMIZER.LR = args.lr
            cfgs.OPTIMIZATION.SCHEDULER.MAX_LR = args.lr
        elif 'OPTIMIZATION' in cfgs:
            print("Found existing OPTIMIZATION section, adding LR to it.")
            cfgs.OPTIMIZATION.OPTIMIZER.LR = args.lr
            cfgs.OPTIMIZATION.SCHEDULER.MAX_LR = args.lr
        else:
            print("Expected OPTIMIZATION section not found in config, creating it with LR.")
            cfgs.OPTIMIZATION = EasyDict({'OPTIMIZER': {'LR': args.lr}, 'SCHEDULER': {'MAX_LR': args.lr}})

    if args.batch_size is not None:
        if 'OPTIMIZATION' in cfgs and hasattr(cfgs.OPTIMIZATION, 'BATCH_SIZE_PER_GPU'):
            cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU = args.batch_size
        elif 'OPTIMIZATION' in cfgs:
            cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU = args.batch_size
        else:
            cfgs.OPTIMIZATION = EasyDict({'BATCH_SIZE_PER_GPU': args.batch_size})

    if args.override_epoch is not None:
        if 'OPTIMIZATION' in cfgs and hasattr(cfgs.OPTIMIZATION, 'NUM_EPOCHS'):
            cfgs.OPTIMIZATION.NUM_EPOCHS = args.override_epoch
        elif 'OPTIMIZATION' in cfgs:
            cfgs.OPTIMIZATION.NUM_EPOCHS = args.override_epoch
        else:
            cfgs.OPTIMIZATION = EasyDict({'NUM_EPOCHS': args.override_epoch})


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
    
    args.output_dir = str(os.path.join(args.save_root_dir, args.exp_group_path, args.tag, args.extra_tag))

    return args, cfgs