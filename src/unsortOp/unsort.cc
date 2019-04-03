#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op_kernel.h>

#include <iostream>

using namespace tensorflow;

REGISTER_OP("Unsort")
.Input("inputs: float") // (batch_size, in_dim)
.Input("indices: int32") // (in_dim, out_dim)
.Input("mean: float") // (in_dim, )
.Output("output: float") // (batch_size, out_dim)
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    tensorflow::shape_inference::ShapeHandle output;
    auto batch_size = c->Dim(c->input(0), 0);
    auto out_dim = c->Dim(c->input(1), 1);
    output = c->Matrix(batch_size, out_dim);
    c->set_output(0, output);
    return Status::OK();
});


class UnsortOp : public OpKernel {
public:
  explicit UnsortOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    std::cout << ">>>> HELLO" << std::endl;
    std::cout << ">>>> INPUT SHAPE" << input_tensor.shape().dims() << std::endl;
    auto output_shape = TensorShape();
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                       &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};

            REGISTER_KERNEL_BUILDER(Name("Unsort").Device(DEVICE_CPU), UnsortOp);
