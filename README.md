
# LightFusion (workname oneStereo)

This is a repository forked from original [OpenStereo](https://github.com/XiandaGuo/OpenStereo) framework for development and evaluation of neural networks for stereo depth perception.

The purpose is:
- add capability to develop fusion models
- present a strong fusion model that can possibly have a faster inference rate than single-modality model
- provide tools for temporal benchmarking

## Example use:

EXPERIMENT_ID="68"
EXPERIMENT_NAME="exp_68_half"
CFG_FILE="./cfgs/onestereo/68_one_stereo_s_sceneflow_lightstereo_MobNetV3_small_fusion.yaml"

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





## Getting Started

Please see [0.get_started.md](docs/0.get_started.md). We also provide the following tutorials for your reference:
- [Prepare dataset](docs/2.prepare_dataset.md)
- [Detailed configuration](docs/3.detailed_config.md)
- [Customize model](docs/4.how_to_create_your_model.md)
- [Advanced usages](docs/5.advanced_usages.md) 

## Acknowledgement
[AANet](https://github.com/haofeixu/aanet) &nbsp; [ACVNet](https://github.com/gangweiX/ACVNet) &nbsp; [CascadeStereo](https://github.com/alibaba/cascade-stereo) &nbsp; [CFNet](https://github.com/gallenszl/CFNet) &nbsp; [COEX](https://github.com/antabangun/coex) &nbsp; [DenseMatching](https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark) &nbsp; [FADNet++](https://github.com/HKBU-HPML/FADNet/tree/fadnet-pp) &nbsp; [GwcNet](https://github.com/xy-guo/GwcNet) &nbsp; [MSNet](https://github.com/cogsys-tuebingen/mobilestereonet) &nbsp; [PSMNet](https://github.com/JiaRenChang/PSMNet) &nbsp; [RAFT](https://github.com/princeton-vl/RAFT-Stereo) &nbsp; [STTR](https://github.com/mli0603/stereo-transformer) &nbsp; [OpenGait](https://github.com/ShiqiYu/OpenGait) &nbsp; [IGEV](https://github.com/gangweiX/IGEV/tree/main/IGEV-Stereo) &nbsp; [NMRF](https://github.com/aeolusguan/NMRF) &nbsp;

## Citation

**Note**: This code is only used for academic purposes, people cannot use this code for anything that might be considered commercial use.
