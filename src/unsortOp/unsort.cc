#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Unsort")
.Input("indices: int32")
.Input("curve: float")
.Output("output: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
              c->set_output(0, c->input(0));
              return Status::OK();
            });
