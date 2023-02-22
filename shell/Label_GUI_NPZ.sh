#!/bin/sh

root=/home/alex/SECEDER
mask_folder=$root/data/mask
image_folder=$root/data/images
feature_folder=$root/data/feature
outf=$root/data/label

script=$root/src/Label_GUI.py

conda activate seceder
python  $script --mask_folder $mask_folder --image_folder $image_folder --feature_folder $feature_folder --outf $outf
