#!/bin/sh

python train.py --gpu_list=1 --input_size=512 --batch_size=12 --nb_workers=6 --training_data_path=../../data/ICDAR2015/train_data/ --validation_data_path=../../data/MLT/val_data_latin/ --checkpoint_path=tmp/icdar2015_east_resnet50/


venv/Scripts/python.exe train.py --gpu_list=0 --input_size=512 --batch_size=1 --nb_workers=5 --training_data_path=./training_samples/ --validation_data_path=./validation_samples/ --checkpoint_path=./icdar2015_east_resnet50/

venv/Scripts/python.exe eval.py --gpu_list=0 --test_data_path=./training_samples --model_path=./icdar2015_east_resnet50/model-50.h5 --output_dir=./icdar2015_east_resnet50/eval/