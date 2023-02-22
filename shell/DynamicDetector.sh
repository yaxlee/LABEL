#!/bin/sh

work_path=/home/endeleze/Desktop/SECEDER
data_path=$work_path/RAFT/demo-frames1
script=$work_path/src/DynamicDetector.py
src_dir0=$data_path/stuttgart_00_000000_000001_leftImg8bit.png
src_dir1=$data_path/stuttgart_00_000000_000002_leftImg8bit.png
test_n=320

conda activate seceder
python  $script --src_dir0 $src_dir0 --src_dir1 $src_dir1 --data_path $data_path --test_n $test_n