gpu_test:
	primusrun sudo singularity exec --nv -B ${HOME}/code/custom-op-dir:/root/custom-op-dir,${HOME}/code/custom-op-dir/cache-bazel:/root/.cache/bazel ${HOME}/tmp/simg/tensorflow-custom-op-gpu.simg sh -c 'cd custom-op-dir/custom-op; bazel build tensorflow_time_two:python/ops/_time_two_ops.so && cp bazel-bin/tensorflow_time_two/python/ops/_time_two_ops.so ../' && sudo chown vatai:vatai ${HOME}/code/custom-op-dir/_time_two_ops.so
	optirun singularity exec --nv -H ~/code ~/tmp/simg/tensorflow-1.13.1-gpu-py3.simg  python custom_gpu_op.py
