#!/bin/sh
# Build the custom op and run custom.py

OPTIRUN=optirun
make -B -C src/unsortOp/ && LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/ CUDA_VISIBLE_DEVICES=0 $OPTIRUN python src/custom_mini.py
