# @Time    : 2023/8/29 02:01
# @Author  : zhangchenming
import random
import torch
import numpy as np
import cv2
import warnings
from torchvision.transforms.functional import normalize
from PIL import Image
from torchvision.transforms import ColorJitter


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class TransposeImage(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        sample['left'] = sample['left'].transpose((2, 0, 1))
        sample['right'] = sample['right'].transpose((2, 0, 1))
        return sample


class ToTensor(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        for k in sample.keys():
            if isinstance(sample[k], np.ndarray):
                if k == 'super_pixel_label':
                    sample[k] = torch.from_numpy(sample[k].copy()).to(torch.int32)
                elif k in ['occ_mask', 'occ_mask_2']:
                    sample[k] = torch.from_numpy(sample[k].copy()).to(torch.bool)
                else:
                    sample[k] = torch.from_numpy(sample[k].copy()).to(torch.float32)
        return sample


class NormalizeImage(object):
    def __init__(self, config):
        self.mean = config.MEAN
        self.std = config.STD

    def __call__(self, sample):
        sample['left'] = normalize(sample['left'] / 255.0, self.mean, self.std)
        sample['right'] = normalize(sample['right'] / 255.0, self.mean, self.std)
        return sample
    
class NormalizeDepthSource(object):
    """Normalize depth source to [0, 1] range for better neural network training"""
    def __init__(self, config):
        self.max_disp = config.MAX_DISP  # 192 for your case
        
    def __call__(self, sample):
        if 'depth_src_0' in sample and sample['depth_src_0'] is not None:
            # Normalize to [0, 1] range
            # sample['depth_src_0'] = sample['depth_src_0'] / self.max_disp
            # Clip to valid range (handle any outliers)
            sample['depth_src_0'] = torch.clamp(sample['depth_src_0'] / self.max_disp, 0.0, 1.0)
        return sample

class RandomCrop(object):
    def __init__(self, config):
        self.crop_size = config.SIZE
        self.base_size = config.SIZE
        self.y_jitter = config.get('Y_JITTER', False)

    def __call__(self, sample):
        crop_height, crop_width = self.crop_size
        height, width = sample['left'].shape[:2]  # (H, W, 3)
        # crop_height = min(height, crop_height)
        # crop_width = min(width, crop_width)
        if crop_width > width or crop_height > height:
            return sample

        n_pixels = 2 if (self.y_jitter and np.random.rand() < 0.5) else 0
        y1 = random.randint(n_pixels, height - crop_height - n_pixels)
        x1 = random.randint(0, width - crop_width)
        y2 = y1 + np.random.randint(-n_pixels, n_pixels + 1)

        for k in sample.keys():
            if k in ['right']:
                sample[k] = sample[k][y2: y2 + crop_height, x1: x1 + crop_width]
            elif k in ['pos']: #iinet
                sample[k] = sample[k][:, y1: y1 + crop_height, x1: x1 + crop_width]
            else: # disp_pred, left, depth_src_0
                sample[k] = sample[k][y1: y1 + crop_height, x1: x1 + crop_width]

        return sample
    
class CreateAdditionalDepthSource(object):
    def __init__(self, config):
        # the config could accept depth source name, but then we need to let know know other transformers of this name (via global?)
        pass
    def __call__(self, sample):
        sample['depth_src_0'] = sample['disp'].copy()
        return sample


class RandomScale(object):
    def __init__(self, config):
        self.config = config
        self.crop_size = config.SIZE
        self.min_scale = config.MIN_SCALE
        self.max_scale = config.MAX_SCALE
        self.scale_prob = config.SCALE_PROB
        self.stretch_prob = config.STRETCH_PROB

        self.max_stretch = 0.2

    def __call__(self, sample):
        ht, wd = sample['left'].shape[:2]
        min_scale = np.maximum((self.crop_size[0] + 8) / float(ht), (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.scale_prob:
            for k in sample.keys():
                if k in ['left', 'right']:
                    sample[k] = cv2.resize(sample[k], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

                elif k in ['disp', 'disp_right', 'depth_src_0']:
                    sample[k] = cv2.resize(sample[k], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                    sample[k] = sample[k] * scale_x

        return sample


class RandomSparseScale(object):
    def __init__(self, config):
        self.crop_size = config.SIZE
        self.min_scale = config.MIN_SCALE
        self.max_scale = config.MAX_SCALE
        self.scale_prob = config.SCALE_PROB

    def __call__(self, sample):
        ht, wd = sample['left'].shape[:2]
        min_scale = np.maximum((self.crop_size[0] + 1) / float(ht), (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.scale_prob:
            for k in sample.keys():
                if k in ['left', 'right']:
                    sample[k] = cv2.resize(sample[k], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

                elif k in ['disp', 'disp_right', 'depth_src_0']:
                    sample[k] = self.sparse_disp_map_reisze(sample[k], fx=scale_x, fy=scale_y)

        return sample

    @staticmethod
    def sparse_disp_map_reisze(disp, fx=1.0, fy=1.0):
        ht, wd = disp.shape[:2]
        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))
        valid = disp > 0.0
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)
        coords = coords.reshape(-1, 2).astype(np.float32)  # 坐标[w, h]

        disp = disp.reshape(-1).astype(np.float32)
        valid = valid.reshape(-1)
        coords = coords[valid]  # disp > 0 的坐标
        disp = disp[valid]  # disp > 0 的值
        coords = coords * [fx, fy]  # resize 后的坐标
        disp = disp * fx  # risize 后的值

        xx = np.round(coords[:, 0]).astype(np.int32)
        yy = np.round(coords[:, 1]).astype(np.int32)
        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        disp = disp[v]

        resized_disp = np.zeros([ht1, wd1], dtype=np.float32)
        resized_disp[yy, xx] = disp

        return resized_disp


class RandomErase(object):
    def __init__(self, config):
        self.eraser_aug_prob = config.PROB
        self.max_erase_time = config.MAX_TIME
        self.bounds = config.BOUNDS

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']

        ht, wd = img1.shape[:2]
        if 'super_pixel_label' in sample:
            occ_mask_2 = np.zeros((ht, wd), dtype=bool)
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, self.max_erase_time + 1)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(self.bounds[0], self.bounds[1])
                dy = np.random.randint(self.bounds[0], self.bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
                if 'super_pixel_label' in sample:
                    occ_mask_2 = np.zeros((ht, wd), dtype=bool)

        sample['left'] = img1
        sample['right'] = img2
        if 'super_pixel_label' in sample:
            sample['occ_mask_2'] = occ_mask_2
        return sample

class RandomEraseOnDepthSource(object):
    def __init__(self, config):
        self.eraser_aug_prob = config.PROB
        self.erase_ratio = config.get('ERASE_RATIO', 0.1)  # Fraction of pixels to erase
        self.max_erase_iterations = config.get('MAX_ITERATIONS', 3)    # Maximum number of erase operations
        self.bounds = config.get('BOUNDS', [50, 100])      # Min/max size of erase rectangles

    def __call__(self, sample):
        if 'depth_src_0' not in sample or sample['depth_src_0'] is None:
            warnings.warn("stereo_trans.py: Missing depth source: sample['depth_src_0']", RuntimeWarning)
            return sample
            
        depth_src_0 = sample['depth_src_0']
        ht, wd = depth_src_0.shape[:2]
        
        if np.random.rand() < self.eraser_aug_prob:
            # Calculate total pixels to erase based on erase_ratio
            total_pixels = ht * wd
            pixels_to_erase = int(total_pixels * self.erase_ratio)
            
            # Perform random number of erase operations (1 to max_erase_iterations)
            num_erase_ops = np.random.randint(1, self.max_erase_iterations + 1)
            
            for _ in range(num_erase_ops):
                # Random position for erase rectangle
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                
                # Random size of erase rectangle within bounds
                dx = np.random.randint(self.bounds[0], self.bounds[1])
                dy = np.random.randint(self.bounds[0], self.bounds[1])
                
                # Ensure rectangle doesn't go out of bounds
                dx = min(dx, wd - x0)
                dy = min(dy, ht - y0)
                
                # Erase by setting to 0 (invalid depth)
                depth_src_0[y0:y0 + dy, x0:x0 + dx] = 0.0

        sample['depth_src_0'] = depth_src_0
        return sample
    
class RandomNoiseOnDepthSource(object):
    def __init__(self, config):
        self.prob = config.PROB
        self.max_noise_level_perc = config.MAX_NOISE_LEVEL_PERC

    def __call__(self, sample):
        if 'depth_src_0' not in sample or sample['depth_src_0'] is None:
            warnings.warn("stereo_trans.py: Missing depth source: sample['depth_src_0']", RuntimeWarning)
            return sample
            # raise ValueError("stereo_trans.py: Missing depth source: sample['depth_src_0']")
        
        if np.random.rand() < self.prob:
            # Check if it's a tensor or numpy array

            # depth_float = depth_src_0.to(torch.float32)
            depth_float = sample['depth_src_0']
            
            # Generate random noise pattern with same shape as depth map
            # Noise ranges from -max_noise_level_perc to +max_noise_level_perc
            noise_pattern = np.random.uniform(
                -self.max_noise_level_perc, 
                self.max_noise_level_perc, 
                size=depth_float.shape
            ).astype(np.float32)

            
            
            # Apply percentage-based noise: depth * (1 + noise_percentage)
            # This means each pixel gets a different random noise value
            # print(f"randomNoiseOnDepth: depth_float dtype: {depth_float.dtype}")
            # print(f"randomNoiseOnDepth: depth_float type: {depth_float.type()}")
            # print(f"randomNoiseOnDepth: noise_pattern dtype: {noise_pattern.dtype}")
            noisy_depth = depth_float * (1.0 + noise_pattern)
            
            # print(f"randomNoiseOnDepth: noisy_depth dtype: {noisy_depth.dtype}")

            
            # Clip negative values to 0 (invalid depth)
            noisy_depth = np.maximum(noisy_depth, 0.0)
            
            sample['depth_src_0'] = noisy_depth

            # print(f"randomNoiseOnDepth: sample['depth_src_0'] dtype: {sample['depth_src_0'].dtype}")

        return sample

def generate_sample_hash(depth_float):

    h, w = depth_float.shape
    
    # Sample key regions more efficiently
    samples = []
    
    # Corner samples (small regions)
    samples.extend(depth_float[:5, -5:].flatten())           # Top-right
    samples.extend(depth_float[-5:, -5:].flatten())         # Bottom-right
    samples.extend(depth_float[h//2:h//2+5, w//2:w//2+5].flatten())  # Center
    
    # Sparse sampling across the image
    step = max(h//10, w//10, 5)  # Adaptive step size
    samples.extend(depth_float[::step, ::step].flatten())
    
    # Statistical fingerprint
    content_string = (
        f"{depth_float.shape}_"
        f"{depth_float.mean():.6f}_"
        f"{depth_float.std():.6f}_"
        f"{depth_float.sum():.6f}_"
        f"{str(samples[-100:])}"  # Only last 100 samples
    )
    
    return hash(content_string)

# import os
class FixedNoiseOnDepthSource(object):
    """
    Applies consistent noise pattern to depth source during validation.
    Uses a fixed random seed to ensure the same noise pattern is applied
    to the same sample across different epochs.
    """
    def __init__(self, config):
        self.prob = config.PROB
        self.max_noise_level_perc = config.MAX_NOISE_LEVEL_PERC
        self.seed = getattr(config, 'SEED', 42)  # Fixed seed for consistency
        self._noise_cache = {}  # Cache noise patterns per sample

        
        # self.last_idx = 0  # Initialize index counter
        # # Create output directory if it doesn't exist
        # os.makedirs(self.output_dir, exist_ok=True)

    def __call__(self, sample):
        if 'depth_src_0' not in sample or sample['depth_src_0'] is None:
            warnings.warn("stereo_trans.py: Missing depth source: sample['depth_src_0']", RuntimeWarning)
            return sample

        depth_float = sample['depth_src_0']
        
        # Create a unique key for this sample based on its content
        # This ensures the same sample gets the same noise pattern
        sample_key = generate_sample_hash(depth_float)

        if sample_key not in self._noise_cache:
            # print(f"FixedNoise: Creating new pattern for sample hash: {sample_key}")
            # Use fixed seed + sample key to generate consistent random decision and noise
            rng = np.random.RandomState(self.seed + abs(sample_key) % 1000000)
            
            # Make deterministic decision about whether to apply noise for this sample
            if rng.rand() < self.prob:
                noise_pattern = rng.uniform(
                    -self.max_noise_level_perc, 
                    self.max_noise_level_perc, 
                    size=depth_float.shape
                ).astype(np.float32)
            else:
                # Generate zero pattern (no noise) when condition is not satisfied
                noise_pattern = np.zeros(depth_float.shape, dtype=np.float32)
            
            self._noise_cache[sample_key] = noise_pattern

        else:
            # print(f"FixedNoise: Using cached pattern for sample hash: {sample_key}")
            noise_pattern = self._noise_cache[sample_key]
        
        # Apply percentage-based noise: depth * (1 + noise_percentage)
        noisy_depth = depth_float * (1.0 + noise_pattern)
        
        # Clip negative values to 0 (invalid depth)
        noisy_depth = np.maximum(noisy_depth, 0.0)
        
        sample['depth_src_0'] = noisy_depth

        return sample


class FixedEraseOnDepthSource(object):
    """
    Applies consistent erase pattern to depth source during validation.
    Uses a fixed random seed to ensure the same erase rectangles are applied
    to the same sample across different epochs.
    """
    def __init__(self, config):
        self.prob = config.PROB
        self.max_iterations = config.MAX_ITERATIONS
        self.bounds = config.BOUNDS
        self.seed = getattr(config, 'SEED', 42)  # Fixed seed for consistency
        self._erase_cache = {}  # Cache erase patterns per sample

    def __call__(self, sample):
        if 'depth_src_0' not in sample or sample['depth_src_0'] is None:
            warnings.warn("stereo_trans.py: Missing depth source: sample['depth_src_0']", RuntimeWarning)
            return sample

        depth_src_0 = sample['depth_src_0']
        ht, wd = depth_src_0.shape[:2]
        
        # Create a unique key for this sample
        sample_key = generate_sample_hash(depth_src_0)

        # print("FixedErase: Current erase cache keys:")
        # print(self._erase_cache)

        if sample_key not in self._erase_cache:
            # print(f"FixedErase: Creating new pattern for sample hash: {sample_key}")
            # Use fixed seed + sample key to generate consistent erase pattern
            rng = np.random.RandomState(self.seed + abs(sample_key) % 1000000)
            
            # Make deterministic decision about whether to apply erase for this sample
            if rng.rand() < self.prob:
                # Generate fixed number of erases (use max_iterations for consistency)
                # choose a random number between 1 and num_erases
                num_erases = rng.randint(1, self.max_iterations + 1)
                erase_rectangles = []
                
                for _ in range(num_erases):
                    x0 = rng.randint(0, wd)
                    y0 = rng.randint(0, ht)
                    dx = rng.randint(self.bounds[0], self.bounds[1])
                    dy = rng.randint(self.bounds[0], self.bounds[1])
                    
                    # Ensure rectangle doesn't go out of bounds
                    dx = min(dx, wd - x0)
                    dy = min(dy, ht - y0)
                    
                    erase_rectangles.append((y0, y0 + dy, x0, x0 + dx))
            else:
                # Generate empty erase pattern (no erase) when condition is not satisfied
                erase_rectangles = []
            
            self._erase_cache[sample_key] = erase_rectangles
        else:
            # print(f"FixedErase: Using cached pattern for sample hash: {sample_key}")
            erase_rectangles = self._erase_cache[sample_key]
        
        # Apply cached erase rectangles
        for y0, y1, x0, x1 in erase_rectangles:
            depth_src_0[y0:y1, x0:x1] = 0.0

        sample['depth_src_0'] = depth_src_0

        return sample


class StereoColorJitter(object):
    def __init__(self, config):
        self.brightness = list(config.BRIGHTNESS)
        self.contrast = list(config.CONTRAST)
        self.saturation = list(config.SATURATION)
        if isinstance(config.HUE, float):
            config.HUE = [-config.HUE, config.HUE]
        self.hue = list(config.HUE)
        self.asymmetric_color_aug_prob = config.ASYMMETRIC_PROB
        self.color_jitter = ColorJitter(brightness=self.brightness, contrast=self.contrast,
                                        saturation=self.saturation, hue=[x / 3.14 for x in self.hue])

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.color_jitter(Image.fromarray(img1.astype(np.uint8))), dtype=np.uint8)
            img2 = np.array(self.color_jitter(Image.fromarray(img2.astype(np.uint8))), dtype=np.uint8)
        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0).astype(np.uint8)
            image_stack = np.array(self.color_jitter(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        sample['left'] = img1
        sample['right'] = img2

        return sample


class RightTopPad(object):
    def __init__(self, config):
        self.size = config.SIZE

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        th, tw = self.size
        h = min(h, th)  # ensure h is within the bounds of the image
        w = min(w, tw)  # ensure w is within the bounds of the image

        pad_left = 0
        pad_right = tw - w
        pad_top = th - h
        pad_bottom = 0
        # apply pad for left, right, disp image, and occ mask
        for k in sample.keys():
            if k in ['left', 'right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                sample[k] = np.pad(sample[k], pad_width, 'edge')

            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right', 'depth_src_0']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right]])
                sample[k] = np.pad(sample[k], pad_width, 'constant', constant_values=0)

        return sample


class DivisiblePad(object):
    def __init__(self, config):
        self.by = config.BY
        self.mode = config.get('MODE', 'tr')

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        if h % self.by != 0:
            pad_h = self.by - h % self.by
        else:
            pad_h = 0
        if w % self.by != 0:
            pad_w = self.by - w % self.by
        else:
            pad_w = 0

        if self.mode == 'round':
            pad_top = pad_h // 2
            pad_right = pad_w // 2
            pad_bottom = pad_h - (pad_h // 2)
            pad_left = pad_w - (pad_w // 2)
        elif self.mode == 'tr':
            pad_top = pad_h
            pad_right = pad_w
            pad_bottom = 0
            pad_left = 0
        else:
            raise Exception('no DivisiblePad mode')

        # apply pad for left, right, disp image, and occ mask
        for k in sample.keys():
            if k in ['left', 'right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                sample[k] = np.pad(sample[k], pad_width, 'edge')

            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right', 'depth_src_0']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right]])
                sample[k] = np.pad(sample[k], pad_width, 'constant', constant_values=0)

        sample['pad'] = [pad_top, pad_right, pad_bottom, pad_left]
        return sample


class RandomFlip(object):
    def __init__(self, config):
        self.config = config
        self.flip_type = config.FLIP_TYPE
        self.prob = config.PROB

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']
        disp_right = sample['disp_right']
        disp_src_0 = sample.get('disp_src_0', None)

        if np.random.rand() < self.prob and self.flip_type == 'horizontal':  # 水平翻转
            img1 = np.ascontiguousarray(img1[:, ::-1])
            img2 = np.ascontiguousarray(img2[:, ::-1])
            disp = np.ascontiguousarray(disp[:, ::-1] * -1.0)
            if disp_src_0 is not None:
                disp_src_0 = np.ascontiguousarray(disp_src_0[:, ::-1] * -1.0)

        if np.random.rand() < self.prob and self.flip_type == 'horizontal_swap':  # 水平翻转并交换
            tmp = np.ascontiguousarray(img1[:, ::-1])
            img1 = np.ascontiguousarray(img2[:, ::-1])
            disp = np.ascontiguousarray(disp_right[:, ::-1])
            img2 = tmp
            if disp_src_0 is not None:
                disp_src_0 = np.ascontiguousarray(disp_src_0[:, ::-1])

        if np.random.rand() < self.prob and self.flip_type == 'vertical':  # 垂直翻转
            img1 = np.ascontiguousarray(img1[::-1, :])
            img2 = np.ascontiguousarray(img2[::-1, :])
            disp = np.ascontiguousarray(disp[::-1, :])
            if disp_src_0 is not None:
                disp_src_0 = np.ascontiguousarray(disp_src_0[::-1, :])

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = disp
        if disp_src_0 is not None:
            sample['disp_src_0'] = disp_src_0
        return sample


class RightBottomCrop(object):
    def __init__(self, config):
        self.size = config.SIZE

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        crop_h, crop_w = self.size
        crop_h = min(h, crop_h)
        crop_w = min(w, crop_w)

        for k in sample.keys():
            sample[k] = sample[k][h - crop_h:, w - crop_w:]
        return sample


class CropOrPad(object):
    def __init__(self, config):
        self.size = config.SIZE
        self.crop_fn = RightBottomCrop(config)
        self.pad_fn = RightTopPad(config)

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        th, tw = self.size
        if th > h or tw > w:
            sample = self.pad_fn(sample)
        else:
            sample = self.crop_fn(sample)

        return sample


class DownscaleDepthSource(object):
    def __init__(self, config):
        self.ds_factor = config.DS_FACTOR
        self.method = config.METHOD
        
        # Validate method
        assert self.method in ["mean", "nearest", "max"], \
            f"method must be one of ['mean', 'nearest', 'max'], got {self.method}"
        
        # Validate downscale factor
        assert isinstance(self.ds_factor, int) and self.ds_factor > 1, \
            f"DS_FACTOR must be an integer > 1, got {self.ds_factor}"

    def __call__(self, sample):       
        if 'depth_src_0' not in sample or sample['depth_src_0'] is None:
            warnings.warn("stereo_trans.py: Missing depth source: sample['depth_src_0']", RuntimeWarning)
            return sample
            
        depth_src_0 = sample['depth_src_0']
        

        depth_array = depth_src_0
        
        # Get original dimensions
        if depth_array.ndim == 2:
            h, w = depth_array.shape
        elif depth_array.ndim == 3 and depth_array.shape[0] == 1:
            # Handle case where depth has shape (1, H, W)
            depth_array = depth_array.squeeze(0)
            h, w = depth_array.shape
        else:
            raise ValueError(f"Expected depth_src_0 to have shape (H, W) or (1, H, W), got {depth_array.shape}")
        
        # Calculate new dimensions
        new_h = h // self.ds_factor
        new_w = w // self.ds_factor
        
        # Ensure we have valid dimensions
        if new_h == 0 or new_w == 0:
            raise ValueError(f"Downscale factor {self.ds_factor} too large for image size ({h}, {w})")
        
        # Apply downscaling based on method
        if self.method == "mean":
            # Reshape and compute mean over blocks
            # Crop to make dimensions divisible by ds_factor
            crop_h = new_h * self.ds_factor
            crop_w = new_w * self.ds_factor
            cropped = depth_array[:crop_h, :crop_w]
            
            # Reshape to group pixels into blocks and take mean
            reshaped = cropped.reshape(new_h, self.ds_factor, new_w, self.ds_factor)
            downscaled = np.mean(reshaped, axis=(1, 3))
            
        elif self.method == "nearest":
            # Simple nearest neighbor downsampling (take every ds_factor-th pixel)
            downscaled = depth_array[::self.ds_factor, ::self.ds_factor]
            
        elif self.method == "max":
            # Take maximum value in each block
            # Crop to make dimensions divisible by ds_factor
            crop_h = new_h * self.ds_factor
            crop_w = new_w * self.ds_factor
            cropped = depth_array[:crop_h, :crop_w]
            
            # Reshape to group pixels into blocks and take max
            reshaped = cropped.reshape(new_h, self.ds_factor, new_w, self.ds_factor)
            downscaled = np.max(reshaped, axis=(1, 3))
        
        # Update sample
        sample['depth_src_0'] = downscaled
        
        return sample
