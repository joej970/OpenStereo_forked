

# Do it once: Concatenating the Image Pair for a Single Pass Feature Extraction in Stereo Depth Sensing
Welcome to the official repository for "Do it once," a simple, plug-and-play methodology to significantly accelerate deep-learning-based stereo depth inference without compromising accuracy.

### 📖 About the Project
In standard stereo depth estimation models, feature extraction is performed sequentially for the left and right images. This requires model weights to be fetched twice from VRAM, creating a major computational bottleneck, especially on memory-constrained edge devices.

Our approach solves this by concatenating the left and right images into a combined tensor, allowing the feature extraction network to process both images in a single batched pass. By fetching weights just once, we significantly reduce memory transactions and accelerate inference.

### ✨ Key Highlights
  🚀 Significant Speedups: Achieves an average inference acceleration of 10% to 39% depending on the specific model architecture (tested with LightStereo, IINet, and LeanStereo).
  🎯 Zero Accuracy Drop: Fully retains the original End-Point-Error (EPE) accuracy of the baseline models.
  🔌 Plug-and-Play: Implemented as a simple wrapper around existing Feature Extraction modules—no need to alter the core network architecture or re-engineer operators.
  🛠️ Configurable: Includes multiple concatenation strategies (e.g., batch dimension, spatial multi-cut) so you can find the optimal graph structure for your specific deployment hardware (e.g., via TensorRT).


When deployed via ONNX and TensorRT on a consumer-grade GPU, our single-pass extraction yields faster tracking across various architectures with only a marginal increase in transient memory usage. Our method can be applied on any existing or new deep-learning based stereo depth model for instant performance boost.


## The implementation:

[OpenStereo_DoItOnce/stereo/modeling/models/LeanStereo/FeatExtractionWrapper.py](./stereo/modeling/models/LeanStereo/FeatExtractionWrapper.py) contains the implementation of our method. Based on the model configuration setting     

```
BACKBONE_CFGS: 
        CONCAT_LEFT_RIGHT: true # concatenates left and right images along batch before feeding to backbone
        CONCAT_LEFT_RIGHT_ALONG: batch # horizontal, vertical, multicut 
```

different implementation of the forward() method is used. 

## Dataset configuration
[OpenStereo_DoItOnce/data/SceneFlow/sceneflow_hpc_finalpass.yaml](./data/SceneFlow/sceneflow_hpc_finalpass.yaml):

### Specify training samples
The dataset configuration file specifies location of .txt file containing a list of all samples. Example .txt file:
```
driving_finalpass/driving__frames_finalpass_webp/frames_finalpass_webp/15mm_focallength/scene_backwards/slow/left/0224.webp driving_finalpass/driving__frames_finalpass_webp/frames_finalpass_webp/15mm_focallength/scene_backwards/slow/right/0224.webp driving_finalpass/driving__disparity/disparity/15mm_focallength/scene_backwards/slow/left/0224.pfm
```
```
driving_finalpass/driving__frames_finalpass_webp/frames_finalpass_webp/15mm_focallength/scene_backwards/slow/left/0132.webp driving_finalpass/driving__frames_finalpass_webp/frames_finalpass_webp/15mm_focallength/scene_backwards/slow/right/0132.webp driving_finalpass/driving__disparity/disparity/15mm_focallength/scene_backwards/slow/left/0132.pfm
```
```
driving_finalpass/driving__frames_finalpass_webp/frames_finalpass_webp/15mm_focallength/scene_backwards/slow/left/0020.webp driving_finalpass/driving__frames_finalpass_webp/frames_finalpass_webp/15mm_focallength/scene_backwards/slow/right/0020.webp driving_finalpass/driving__disparity/disparity/15mm_focallength/scene_backwards/slow/left/0020.pfm
```
The ```DATA_PATH``` field in dataset configuration file should point to the base dataset location (i.e. DATA_PATH will be prepended to the location specified in the .txt file).

### Specify size
If not training on full dataset (or just debugging), you can limit the number of samples using fields:
```
SUBSET_SIZE_TRAINING: 0.5    # Use 10% of training data
SUBSET_SIZE_VALID: 0.5       # Use 50% of validation data  
SUBSET_SIZE_TEST: 1.0        # Use 100% of test data
SUBSET_SELECTION_SEED: 42  # Seed for reproducibility
```

### Specify number of epochs
If you want to specify number of epochs for a specific model use (only IINet, LeanStereo, LightStereo supported):
```
NUM_EPOCHS: 50
NUM_EPOCHS_IINET: 105 # overrides model config
NUM_EPOCHS_LEANSTEREO: 80 # overrides model config
NUM_EPOCHS_LIGHTSTEREO: 90 # overrides model config
```

Alternatively, you can use option '--override_epoch' when calling python train.py.

### Specify augmentation

Using ```DATA_TRANSFORM```.

### Save error maps

Using 
```
  SAVE_ERROR_MAP_LIST:
    - /d/hpc/home/<user>/datasets/sceneflow/flyingthings3d/frames_finalpass/TEST/C/0134/left/0013.png
    - /d/hpc/home/<user>/datasets/sceneflow/flyingthings3d/frames_finalpass/TEST/C/0036/left/0006.png
```
If this sample name is also specified in the .txt file containing samples and ```TEST_VISUALIZATION``` in model configuration is set to true, then image showing ground truth, prediction and error map is generated.


## Model configuration file (.yaml)
[OpenStereo_DoItOnce/cfgs/LeanStereo/300a_LeanStereo_sceneflow.yaml](./cfgs/LeanStereo/300a_LeanStereo_sceneflow.yaml):

This .yaml is used to configure model. Most of the parameteres are related to a specific model, while other are related to the training configuration. Some more specific ones are:

```
MODEL:
    PRETRAINED_MODEL: '' # this field can be used to load a pretrained model and then start training from beginning.
    # CKPT: -1 # default; starts training from beggining
    CKPT_DIR: ''

    CKPT_DIR: '/d/hpc/home/<user>/OpenStereo_DoItOnce/output/SceneFlowDataset/LeanStereo/300_LeanStereo_sceneflow/300_exp_300_half_65200718/ckpt/' # checkpoint location; contains checkpoint_epoch_78.pth, best_model.pth, 
    CKPT: 39 # in case training is done in several stages, this can be used to resume from a checkpoint (loads checkpoint_epoch_<nr>.pth)

    BACKBONE_CFGS: 
      CONCAT_LEFT_RIGHT: true # concatenates left and right images along batch before feeding to backbone
      CONCAT_LEFT_RIGHT_ALONG: batch # horizontal, vertical, multicut 

OPTIMIZATION:  
    OPTI_REINIT: true # in case of loading from a checkpoint, reinitialize the optimizer to follow new optimizer configs. This can be usefull if optimizer settings have changed or optimizer settings were not saved in the checkpoint file. Set this to true if getting error: 'ValueError: Tried to step 8386 times. The specified number of total steps is 8385.'

    # original
    NUM_EPOCHS: 320 # set number of training epochs. This is overriden by dataset configuartion file (sceneflow_hpw_finalpass.yaml), which is in turn overriden by '--override_epoch' argument to python train.py

```

## Training

### How to use (if using HPC and slurm management and job scheduling system)
If using slurm to launch processes, refer to our example slurm script to train, evaluate using TensorRT and parse detailed training logs into a single more succinct log.

Example slurm script: [OpenStereo_DoItOnce/slurm_training/300_half_train_v100s_wn.slurm](./slurm_training/300_half_train_v100s_wn.slurm).

### If not using slurm
Call the same command as specified in the above mentioned slurm script:
```

CFG_FILE="./cfgs/LeanStereo/300_LeanStereo_sceneflow.yaml"
DATASET_CFG_FILE="./data/SceneFlow/sceneflow_hpc_finalpass_dev.yaml"
WORKERS=10 # number of cpu cores
EXPERIMENT_ID=300 # used to generate output location and filenames
EXPERIMENT_NAME="exp_300"
JOB_ID=123456 # used to generate output location and filenames
NODEID="any_name"

python ./tools/train.py \
  --force_override \
  --cfg_file $CFG_FILE \
  --data_cfg_file $DATASET_CFG_FILE \
  --fix_random_seed \
  --workers $WORKERS \
  --pin_memory \
  --force_override \
  --save_root_dir ./output \
  --experiment_id ${EXPERIMENT_ID} \
  --slurm_job_id ${JOB_ID} \
  --extra_tag ${EXPERIMENT_ID}_${EXPERIMENT_NAME}_${JOB_ID} | tee ../output/${JOB_ID}_${EXPERIMENT_ID}_${EXPERIMENT_NAME}_${NODEID}_half.log
```


### Training: train.py:
`python ./tools/train.py \
  --force_override \
  --backend $BACKEND \
  --cfg_file $CFG_FILE \
  --data_cfg_file ./data/SceneFlow/sceneflow_hpc_finalpass.yaml \
  --fix_random_seed \
  --workers $SLURM_CPUS_PER_TASK \
  --pin_memory \
  --force_override \
  --save_root_dir ./output \
  --experiment_id ${EXPERIMENT_ID} \
  --slurm_job_id ${SLURM_JOB_ID} \
  --extra_tag ${EXPERIMENT_ID}_${EXPERIMENT_NAME}_${SLURM_JOB_ID}`

### Temporal benchmarking using ONNX
  `python ./tools/onnx_evaluation_tensorRT.py \
  --cfg_file $CFG_FILE \
  --data_cfg_file ./data/SceneFlow/sceneflow_hpc_finalpass.yaml \
  --workers $SLURM_CPUS_PER_TASK \
  --pin_memory \
  --save_root_dir ./output \
  --experiment_id ${EXPERIMENT_ID} \
  --slurm_job_id ${SLURM_JOB_ID} \
  --extra_tag ${EXPERIMENT_ID}_${EXPERIMENT_NAME}_${SLURM_JOB_ID} | tee ../slurm_output/${SLURM_JOB_ID}_${EXPERIMENT_ID}_${EXPERIMENT_NAME}_${SLURM_NODEID}_half.log`

### End result parsing: Reads the log file and extract data such as GFLOPS, best epoch EPE, test EPE and inference rate using TensorRT
`python ./tools/parse_training_log.py \
  --cfg_file $CFG_FILE \
  --data_cfg_file ./data/SceneFlow/sceneflow_hpc_finalpass.yaml \
  --workers $SLURM_CPUS_PER_TASK \
  --pin_memory \
  --save_root_dir ./output \
  --experiment_id ${EXPERIMENT_ID} \
  --slurm_job_id ${SLURM_JOB_ID} \
  --extra_tag ${EXPERIMENT_ID}_${EXPERIMENT_NAME}_${SLURM_JOB_ID} | tee ../slurm_output/${SLURM_JOB_ID}_${EXPERIMENT_ID}_${EXPERIMENT_NAME}_${SLURM_NODEID}_half.log`

## Citation

TBD

## Acknowledgement
This is a repository forked from original [OpenStereo](https://github.com/XiandaGuo/OpenStereo) framework for development and evaluation of neural networks for stereo depth perception.

### Getting Started (from original OpenStereo repository)

Please see [0.get_started.md](docs/0.get_started.md). We also provide the following tutorials for your reference:
- [Prepare dataset](docs/2.prepare_dataset.md)
- [Detailed configuration](docs/3.detailed_config.md)
- [Customize model](docs/4.how_to_create_your_model.md)
- [Advanced usages](docs/5.advanced_usages.md) 



**Note**: This code is only used for academic purposes, people cannot use this code for anything that might be considered commercial use.
