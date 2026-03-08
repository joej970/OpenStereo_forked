# @Time    : 2023/8/28 22:28
# @Author  : zhangchenming
import os
import yaml
import random
import shutil
import numpy as np
import torch
import logging
import inspect
from easydict import EasyDict
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt
import cv2


def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    return src_cfgs


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_valid_args(obj, input_args, free_keys=None):
    if free_keys is None:
        free_keys = []
    if inspect.isfunction(obj):
        expected_keys = inspect.getfullargspec(obj)[0]
    elif inspect.isclass(obj):
        expected_keys = inspect.getfullargspec(obj.__init__)[0]
    else:
        raise ValueError('Just support function and class object!')
    unexpect_keys = list()
    expected_args = {}
    for k, v in input_args.items():
        k = k.lower()
        if k == 't_max':
            k = 'T_max'
        if k in expected_keys:
            print(f"Found expected key: {k}")
            expected_args[k] = v
        elif k in free_keys:
            print(f"Found free key: {k}")
            pass
        else:
            unexpect_keys.append(k)
    if unexpect_keys:
        print("Find Unexpected Args(%s) in the Configuration of - %s -" % (', '.join(unexpect_keys), obj.__name__))
    return expected_args


def backup_source_code(backup_dir):
    # 子文件夹下的同名也会被忽略
    ignore_hidden = shutil.ignore_patterns(
        ".idea", ".git*", "*pycache*",
        "cfgs", "data", "output")

    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

    shutil.copytree('.', backup_dir, ignore=ignore_hidden)
    # os.system("chmod -R g+w {}".format(backup_dir))


def log_configs(cfgs, pre='cfgs', logger=None):
    for key, val in cfgs.items():
        if isinstance(cfgs[key], EasyDict):
            logger.info('----------- %s -----------' % key)
            log_configs(cfgs[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def save_checkpoint(model, optimizer, scheduler, scaler, is_dist, epoch, filename='checkpoint'):
    if is_dist:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    optim_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    scaler_state = scaler.state_dict()

    state = {'epoch': epoch,
             'model_state': model_state,
             'optimizer_state': optim_state,
             'scheduler_state': scheduler_state,
             'scaler_state': scaler_state}
    torch.save(state, filename)


def freeze_bn(module):
    """Freeze the batch normalization layers."""
    for m in module.modules():
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    return module


# def convert_state_dict(ori_state_dict, is_dist=True):
#     new_state_dict = OrderedDict()
#     if is_dist:
#         if not next(iter(ori_state_dict)).startswith('module'):
#             for k, v in ori_state_dict.items():
#                 new_state_dict[f'module.{k}'] = v
#         else:
#             new_state_dict = ori_state_dict
#     else:
#         if not next(iter(ori_state_dict)).startswith('module'):
#             new_state_dict = ori_state_dict
#         else:
#             for k, v in ori_state_dict.items():
#                 k = k.replace('module.', '')
#                 new_state_dict[k] = v
#
#     return new_state_dict


def load_params_from_file(model, filename, device, dist_mode, logger, strict=True):
    checkpoint = torch.load(filename, map_location=device)
    pretrained_state_dict = checkpoint['model_state']
    tmp_model = model.module if dist_mode else model
    state_dict = tmp_model.state_dict()

    unused_state_dict = {}
    update_state_dict = {}
    unupdate_state_dict = {}
    for key, val in pretrained_state_dict.items():
        if key in state_dict and state_dict[key].shape == val.shape:
            update_state_dict[key] = val
        else:
            unused_state_dict[key] = val
    for key in state_dict:
        if key not in update_state_dict:
            unupdate_state_dict[key] = state_dict[key]

    if strict:
        tmp_model.load_state_dict(update_state_dict)
    else:
        state_dict.update(update_state_dict)
        tmp_model.load_state_dict(state_dict)

    message = 'Unused weight: '
    for key, val in unused_state_dict.items():
        message += str(key) + ':' + str(val.shape) + ', '
    if logger:
        logger.info(message)
    else:
        print(message)

    message = 'Not updated weight: '
    for key, val in unupdate_state_dict.items():
        message += str(key) + ':' + str(val.shape) + ', '
    if logger:
        logger.info(message)
    else:
        print(message)

# return a 1D tensor (array) of line indices that are all zero
def find_all_zero_lines(tensor):
    # tensor: [h, w]
    zero_mask = (tensor == 0)
    zero_lines = np.all(zero_mask, axis=-1)
    zero_lines_indices = np.nonzero(zero_lines)[0]
    # print(f"Found {len(zero_lines_indices)} all-zero lines. Indices: {zero_lines_indices}")
    return zero_lines_indices

# returns a 1D tensor (array) of line indices that have at least one non-zero element
def find_nonzero_lines(tensor):
    # tensor: [h, w]
    nonzero_mask = (tensor != 0)
    nonzero_lines = torch.any(nonzero_mask, axis=-1)
    nonzero_lines = torch.nonzero(nonzero_lines)[0]
    # print(f"Found {len(nonzero_lines)} non-zero lines. Indices: {nonzero_lines}")
    return nonzero_lines

# Given a 1D tensor of line indices, return a new tensor that only includes the longest sequence of consecutive line indices starting from the first index. For example, if the input is [0, 1, 2, 4, 5], the output should be [0, 1, 2] because the sequence breaks at index 3. 
def keep_continous_lines(line_indices):
    for l in range(len(line_indices)-1):
        if line_indices[l+1] - line_indices[l] > 1:
            return line_indices[:l+1]
    return line_indices

def get_last_all_zero_line(line_indices):
    if len(line_indices) == 0:
        return None
    return line_indices[-1]

def get_first_nonzero_line(tensor):
    zero_lines = find_all_zero_lines(tensor)
    zero_lines = keep_continous_lines(zero_lines)
    last_zero_line = get_last_all_zero_line(zero_lines)
    first_nonzero_line = last_zero_line + 1 if last_zero_line is not None else 0
    # print(f"First nonzero line index: {first_nonzero_line}")
    return first_nonzero_line

def trim_tensor_by_line_indices(tensor, first_nonzero_line):
    # print(f"Trimming tensor by line indices. Keeping img from First nonzero line onward: {first_nonzero_line}")
    return tensor[..., first_nonzero_line:, :]


def color_map_tensorboard(disp_gt, pred, disp_max=192):
    cm = plt.get_cmap('plasma')

    disp_gt = disp_gt.detach().data.cpu().numpy()
    pred = pred.detach().data.cpu().numpy()

    first_nonzero_line = get_first_nonzero_line(disp_gt)
    disp_gt = trim_tensor_by_line_indices(disp_gt, first_nonzero_line)
    pred = trim_tensor_by_line_indices(pred, first_nonzero_line)

    error_map = np.abs(pred - disp_gt)

    disp_gt = np.clip(disp_gt, a_min=0, a_max=disp_max)
    pred = np.clip(pred, a_min=0, a_max=disp_max)

    gt_tmp = 255.0 * disp_gt / disp_max
    pred_tmp = 255.0 * pred / disp_max
    error_map_tmp = 255.0 * error_map / np.max(error_map)

    gt_tmp = cm(gt_tmp.astype('uint8'))
    pred_tmp = cm(pred_tmp.astype('uint8'))
    error_map_tmp = cm(error_map_tmp.astype('uint8'))

    gt_tmp = np.transpose(gt_tmp[:, :, :3], (2, 0, 1))
    pred_tmp = np.transpose(pred_tmp[:, :, :3], (2, 0, 1))
    error_map_tmp = np.transpose(error_map_tmp[:, :, :3], (2, 0, 1))

    color_disp_c = np.concatenate((gt_tmp, pred_tmp, error_map_tmp), axis=1)
    color_disp_c = torch.from_numpy(color_disp_c)

    return color_disp_c


def write_tensorboard(tb_writer, tb_info, step):
    for k, v in tb_info.items():
        module_name = k.split('/')[0]
        writer_module = getattr(tb_writer, 'add_' + module_name)
        board_name = k.replace(module_name + "/", '')
        v = v.detach() if torch.is_tensor(v) else v
        if module_name == 'image' and v.dim() == 2:
            writer_module(board_name, v, step, dataformats='HW')
        else:
            writer_module(board_name, v, step)


def draw_depth_2_image(disp, image, baseline=0.54, focallength=1.003556e+3):
    # baseline 单位米
    disp = disp.detach().data.cpu().numpy()  # [h, w]
    image = image.detach().data.cpu().numpy()  # [3, h, w]
    image = np.transpose(image, [1, 2, 0])  # [h, w, 3]
    image = np.ascontiguousarray(image, dtype=np.uint8)

    point_size = 2
    point_colors = [(255, 255, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    thickness = -1

    depths = baseline * focallength / disp
    depths[np.isinf(depths)] = 0.0
    depths[np.isnan(depths)] = 0.

    print('depth', depths.shape)
    print('image', image.shape)
    assert depths.shape[:2] == image.shape[:2]

    for i in range(depths.shape[0]):
        for j in range(depths.shape[1]):
            if depths[i, j] > 40:
                point_color = point_colors[-1]
            elif depths[i, j] > 20:
                point_color = point_colors[-2]
            elif depths[i, j] > 15:
                point_color = point_colors[-3]
            elif depths[i, j] > 8:
                point_color = point_colors[-4]
            elif depths[i, j] > 0.5:
                point_color = point_colors[-5]
            else:
                continue
            cv2.circle(image, (j, i), point_size, point_color, thickness)

    image = np.transpose(image, [2, 0, 1])
    image = np.ascontiguousarray(image, dtype=np.float32)
    return torch.from_numpy(image)

def get_pos_fullres(fx, w, h):
    x_range = (np.linspace(0, w - 1, w) + 0.5 - w // 2) / fx
    y_range = (np.linspace(0, h - 1, h) + 0.5 - h // 2) / fx
    x, y = np.meshgrid(x_range, y_range)
    z = np.ones_like(x)
    pos_grid = np.stack([x, y, z], axis=0).astype(np.float32)
    return pos_grid

def get_filename_from_path(full_path: str) -> str:
    """
    Trim everything before the first occurrence of dataset_name (case-insensitive),
    then replace path separators with '-' to create a filename-friendly string.
    If dataset_name is not found, the full path (normalized) is used.
    """
    # normalize to forward slashes to make searching consistent across platforms
    norm = full_path.replace('\\', '/')
    idx = norm.lower().find('datasets')
    if idx != -1: # if 'datasets' found
        idx = idx + len('datasets/')  # move index to the end of 'datasets'
    sub = norm[idx:] if idx != -1 else norm # skip 'datasets/' part
    sub = sub.strip('/')               # remove any leading/trailing slashes
    # remove extension of the final path component
    sub = str(Path(sub).with_suffix(''))
    return sub.replace('/', '-')       # replace remaining separators with '-'

def save_tensor_as_png(tensor: torch.Tensor, path: str, png_compression: int = 3):
    """
    Save a torch tensor as PNG using OpenCV (no PIL).
    - tensor: CHW or HWC, float (0-1 or 0-255) or uint8
    - path: output file path (should end with .png)
    - png_compression: 0 (no) .. 9 (max compression)
    """
    img = tensor.detach().cpu()

    # convert CHW -> HWC if needed (common for PyTorch)
    if img.dim() == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)

    img = img.contiguous().numpy()

    # convert floats -> uint8 (supports [0,1] or [0,255])
    if np.issubdtype(img.dtype, np.floating):
        if img.max() <= 1.0:
            img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            img = np.clip(img, 0.0, 255.0).round().astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # OpenCV expects BGR for color images
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)])

def get_n_max_values(input_tensor: torch.Tensor, n: int, valid_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Extracts the n maximum values from an input tensor of any shape.
    
    Args:
        input_tensor (torch.Tensor): The input tensor of any shape.
        n (int): The number of maximum values to extract.
        valid_mask (torch.Tensor, optional): A boolean mask of the same shape as input_tensor.
                                             True indicates a valid value. Defaults to None.

    Returns:
        torch.Tensor: A 1D tensor containing the top n values sorted descending.
                      If fewer than n valid values exist, returns all of them.
    """
    # 1. Flatten the input to treat it as a single pool of numbers
    flat_input = input_tensor.detach().flatten()
    
    if valid_mask is not None:
        flat_mask = valid_mask.detach().flatten().bool()
        # 2. Filter: Keep only the elements where mask is True
        valid_values = flat_input[flat_mask]
    else:
        valid_values = flat_input
        
    # 3. Safety check: ensure we don't ask for more values than exist
    k = min(n, valid_values.numel())
    
    if k == 0:
        return torch.tensor([], device=input_tensor.device, dtype=input_tensor.dtype)
        
    sorting_values = valid_values.clone()
    nan_mask = torch.isnan(sorting_values)
    if nan_mask.any():
        sorting_values[nan_mask] = 969696.0 #

    # 4. Find the top k values
    # torch.topk is generally faster than sorting for small k
    values, _ = torch.topk(valid_values, k, sorted=True)
    
    return values

def check_max_vals(x, name : str = 'unknown', n_max_vals : int = 5):
    max_vals = get_n_max_values(x, n_max_vals)
    text = f"Max values of {name}: {max_vals}"
    return text


def draw_train_loss_lr(output_dir, job_id, experiment_id, train_epochs, 
               train_losses = None, train_lrs = None, eval_epochs = None, eval_epes = None):

    if not isinstance(output_dir, str):
        output_dir = str(output_dir)

    # Plot 1: Training Loss and Learning Rate
    fig, ax1 = plt.subplots(figsize=(12, 6), layout='constrained')
    
    # Loss on right axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Loss', color=color)
    ax2.plot(train_epochs, train_losses, color=color, marker='o', label='Training Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(True, alpha=0.3)
    
    # Set y-axis limit for loss to 1.5 times the second epoch value (epoch 1)
    if len(train_losses) > 1 and not np.isnan(train_losses[1]) and train_losses[1] > 0:
        max_loss_display = 1.5 * train_losses[1]
        lim_down = np.nanmin(train_losses) * 0.8 if train_losses else 0.0
        ax2.set_ylim(bottom=lim_down, top=max_loss_display)
    
    # Evaluation epe on right axis (if available)
    if eval_epochs and eval_epes:
        ax3 = ax1.twinx()
        color = 'tab:orange'
        ax3.set_ylabel('EPE', color=color)
        ax3.plot(eval_epochs, eval_epes, color=color, marker='^', label='Evaluation EPE')
        ax3.tick_params(axis='y', labelcolor=color)
        lim_up = np.nanmax(eval_epes) * 1.5
        lim_down = np.nanmin(eval_epes) * 0.8
        ax3.set_ylim(bottom=lim_down, top=lim_up)
    # else:
    #     print(f"No evaluation EPE data provided, skipping EPE plot on training progress graph. eval_epochs: {eval_epochs}, eval_epes: {eval_epes}")

    # Learning rate on left axis
    color = 'tab:blue'
    ax1.set_ylabel('Learning Rate', color=color)
    ax1.plot(train_epochs, train_lrs, color=color, marker='s', label='Learning Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')  # Log scale for LR
    
    plt.title(f'Training Progress: Loss and Learning Rate (Job {job_id}, Exp {experiment_id})')
    plt.tight_layout()
    plt.savefig(output_dir + f'/{job_id}_{experiment_id}_training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}")
    print(f"  - {job_id}_{experiment_id}_training_progress.png")



def draw_eval_epe(output_dir, job_id, experiment_id, eval_epochs = None, eval_epes = None
               ):
    
    if not isinstance(output_dir, str):
        output_dir = str(output_dir)

    # Plot 2: Evaluation EPE
    plt.figure(figsize=(10, 6))
    plt.plot(eval_epochs, eval_epes, color='tab:green', marker='o', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('EPE (End Point Error)')
    plt.title(f'Evaluation EPE Over Training (Job {job_id}, Exp {experiment_id})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir + f'/{job_id}_{experiment_id}_evaluation_epe.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - {job_id}_{experiment_id}_evaluation_epe.png")

