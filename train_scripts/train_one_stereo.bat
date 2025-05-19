echo hello
python -c "import sys; print(sys.executable)"
python .\tools\train.py --cfg_file .\cfgs\onestereo\one_stereo_s_sceneflow_dev.yaml --data_cfg_file .\cfgs\onestereo\one_stereo_s_sceneflow_asus_config.yaml --force_override