#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op_kernel.h>

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
    const int64 batch_size = inputs_tensor.dim_size(0);
    auto inputs_matrix = inputs_tensor.matrix<float>();

    const Tensor& indices_tensor = context->input(1);
    const int64 in_dim = indices_tensor.dim_size(0);
    const int64 out_dim = indices_tensor.dim_size(1);
    auto indices_matrix = indices_tensor.matrix<int32>();

    const Tensor& mean_tensor = context->input(2);
    auto mean_flat = mean_tensor.flat<float>();

    Tensor* outputs_tensor = NULL;
    const TensorShape outputs_shape = TensorShape({batch_size, out_dim});
    OP_REQUIRES_OK(context, context->allocate_output(0, outputs_shape,
                                                     &outputs_tensor));
    auto outputs_matrix = outputs_tensor->matrix<float>();

    // Initial performance thoughts:
    // - Do we need to clear the allocated outputs_tensor?
    // - Is the memory accessed in the correct pattern?

    // Clear outputs_tensor
    for (size_t sample_idx = 0; sample_idx < batch_size; sample_idx++)
      for (size_t i = 0; i < out_dim; i++)
        outputs_matrix(sample_idx, i) = 0;

    for (size_t sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      for (size_t i = 0; i < in_dim; i++) { // in_dim == 3
        for (size_t j = 0; j < out_dim; j++) { // out_dim == 2
          const float prod = inputs_matrix(sample_idx, i) *
                             mean_flat(indices_matrix(i, j));
          outputs_matrix(sample_idx, j) += prod;
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Unsort").Device(DEVICE_CPU), UnsortOp);
