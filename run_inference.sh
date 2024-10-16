#!bin/bash

python model_inference.py -config configs/sen1floods11_config.py -ckpt sen1floods11_Prithvi_100M.pth -input input/ -output output/ -input_type tif -bands 0 1 2 3 4 5