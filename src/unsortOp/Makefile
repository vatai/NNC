CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

UNSORT_SRCS = $(wildcard *.cc) $(wildcard *.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

UNSORT_TARGET_LIB = unsort_ops.so
UNSORT_GPU_ONLY_TARGET_LIB = unsort_ops.cu.o

# unsort op for CPU
unsort_op: $(UNSORT_TARGET_LIB)

$(UNSORT_TARGET_LIB): $(UNSORT_SRCS) $(UNSORT_GPU_ONLY_TARGET_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/opt/cuda/targets/x86_64-linux/include -L/opt/cuda/targets/x86_64-linux/lib -lcudart

$(UNSORT_GPU_ONLY_TARGET_LIB): unsort.cu.cc
	$(NVCC) -std=c++11 -c -o $@ $^  $(TF_LFLAGS) -D GOOGLE_CUDA=1 -I/usr/lib/python3.7/site-packages/tensorflow/include -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr
