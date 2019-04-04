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
    const Tensor& inputs_tensor = context->input(0);
    const Tensor& indices_tensor = context->input(1);
    const Tensor& mean_tensor = context->input(2);

    //// auto input = input_tensor.flat<float>();

    std::cout << ">>>> input tensors created" << std::endl;

    const int64 batch_size = inputs_tensor.dim_size(0);
    const int64 out_dim = indices_tensor.dim_size(1);

    std::cout << ">>>>> batch_size, out_dim: "
              << batch_size << ", "
              << out_dim << std::endl;

    // Create an output tensor
    Tensor* output_tensor = NULL;

    const TensorShape output_shape = TensorShape({batch_size, out_dim});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
    std::cout << ">>>> output allocated" << std::endl;
    //// auto output_flat = output_tensor->flat<float>();
    

    std::cout << ">>>> DONE" << std::endl;
  }
};

REGISTER_KERNEL_BUILDER(Name("Unsort").Device(DEVICE_CPU), UnsortOp);
