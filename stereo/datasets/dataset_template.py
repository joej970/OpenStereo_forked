# @Time    : 2023/11/9 16:50
# @Author  : zhangchenming
import os
import random
import torch.utils.data as torch_data
from .dataset_utils import stereo_trans


def build_transform_by_cfg(transform_config):
    transform_compose = []
    for cur_cfg in transform_config:
        cur_augmentor = getattr(stereo_trans, cur_cfg.NAME)(config=cur_cfg)
        transform_compose.append(cur_augmentor)
    return stereo_trans.Compose(transform_compose)


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__()
        self.data_info = data_info
        self.data_cfg = data_cfg
        self.mode = mode
        self.root = self.data_info.DATA_PATH

        self.split_file = self.data_info.DATA_SPLIT[self.mode.upper()]
        self.data_list = []
        if os.path.exists(self.split_file):
            with open(self.split_file, 'r') as fp:
                self.data_list.extend([x.strip().split(' ') for x in fp.readlines()])
        else:
            split_file_expanded = os.path.expanduser(self.data_info.DATA_SPLIT[self.mode.upper()])
            print(f"Expanded filename '{self.split_file}' to '{split_file_expanded}'.")
            if os.path.exists(split_file_expanded):
                with open(split_file_expanded, 'r') as fp:
                    self.data_list.extend([x.strip().split(' ') for x in fp.readlines()])
            else:
                raise FileNotFoundError(f"Neither file {self.split_file} nor {split_file_expanded} exist.")

        # Apply mode-specific SUBSET_SIZE if specified
        subset_size = None
        if self.mode.lower() == 'training':
            subset_size = getattr(self.data_cfg, 'SUBSET_SIZE_TRAINING', None)
        elif self.mode.lower() == 'evaluating':
            subset_size = getattr(self.data_cfg, 'SUBSET_SIZE_VALID', None)
        elif self.mode.lower() == 'testing':
            subset_size = getattr(self.data_cfg, 'SUBSET_SIZE_TEST', None)
        
        # Take a random subset of the data if specified
        if subset_size is not None:
            original_size = len(self.data_list)
            subset_length = int(original_size * subset_size)
            
            # Save current random state
            current_random_state = random.getstate()
            
            try:
                # Set seed for reproducible subset selection
                subset_seed = getattr(self.data_cfg, 'SUBSET_SELECTION_SEED', 42)
                random.seed(subset_seed)
                
                # Create random indices and select subset
                indices = list(range(original_size))
                selected_indices = random.sample(indices, subset_length)
                selected_indices.sort()  # Sort to maintain some order for debugging
                
                # Select the subset using the random indices
                self.data_list = [self.data_list[i] for i in selected_indices]
                
                print(f"Using {self.mode} random subset: {subset_length}/{original_size} samples ({subset_size*100:.1f}%) with seed {subset_seed}")
                
            finally:
                # Restore original random state
                random.setstate(current_random_state)



        transform_config = self.data_cfg.DATA_TRANSFORM[self.mode.upper()]
        self.transform = build_transform_by_cfg(transform_config)

    def __len__(self):
        return len(self.data_list)


def get_size(base_size, w_range, h_range, random_type):
    if random_type == 'range':
        w = random.randint(int(w_range[0] * base_size[1]), int(w_range[1] * base_size[1]))
        h = random.randint(int(h_range[0] * base_size[0]), int(h_range[1] * base_size[0]))
    elif random_type == 'choice':
        w = random.choice(w_range) if isinstance(w_range, list) else w_range
        h = random.choice(h_range) if isinstance(h_range, list) else h_range
    else:
        raise NotImplementedError
    return int(h), int(w)


def custom_collate(data_list, concat_dataset, batch_uniform=False,
                   random_type=None, h_range=None, w_range=None):
    if batch_uniform:
        for each_dataset in concat_dataset.datasets:
            for cur_t in each_dataset.transform.transforms:
                if type(cur_t).__name__ == 'RandomCrop':
                    base_size = cur_t.base_size
                    cur_t.crop_size = get_size(base_size, w_range, h_range, random_type)
                    break

    return torch_data.default_collate(data_list)
