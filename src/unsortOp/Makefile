CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

# UNSORT_SRCS = $(wildcard tensorflow_unsort/cc/kernels/*.cc) $(wildcard tensorflow_unsort/cc/ops/*.cc)
UNSORT_SRCS = $(wildcard *.cc) $(wildcard *.cc)
## TIME_TWO_SRCS = tensorflow_time_two/cc/kernels/time_two_kernels.cc $(wildcard tensorflow_time_two/cc/kernels/*.h) $(wildcard tensorflow_time_two/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

UNSORT_TARGET_LIB = unsort_ops.so
## TIME_TWO_GPU_ONLY_TARGET_LIB = custOp/_time_two_ops.cu.o
## TIME_TWO_TARGET_LIB = tensorflow_time_two/python/ops/_time_two_ops.so

# unsort op for CPU
unsort_op: $(UNSORT_TARGET_LIB)

$(UNSORT_TARGET_LIB): $(UNSORT_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}
