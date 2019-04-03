make -B -C src/unsortOp/ _unsort_ops.so && LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/ CUDA_VISIBLE_DEVICES=0 optirun python src/custom.py
