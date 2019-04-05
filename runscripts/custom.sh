#!/bin/sh
# Build the custom op and run custom.py.
# Optionally run with $optirun.
# Take care of LD_LIBRARY_PATH bug.
# Take care of multiple GPUs with CUDA_VISIBLE_DEVICES

if  [ -x "$(command -v optirun)" ]; then OPTIRUN=optirun; else OPTIRUN=; fi
OPTIRUN=     # comment this line to DISABLE OPTIRUN
MAKEFLAG=-B  # comment this line not to force RECOMPILATION
GPUs=0       # gpu LIST: 0 or 0,1
make $MAKEFLAG -C src/unsortOp/ && LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/ CUDA_VISIBLE_DEVICES=$GPUs $OPTIRUN python src/custom.py
