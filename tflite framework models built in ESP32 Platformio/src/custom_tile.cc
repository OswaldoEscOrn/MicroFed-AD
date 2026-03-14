#include "custom_tile.h"

#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace custom {
namespace {

constexpr int kInputTensor = 0;
constexpr int kMultiplesTensor = 1;
constexpr int kOutputTensor = 0;

static int32_t Product(const int32_t* dims, int n) {
  int32_t p = 1;
  for (int i = 0; i < n; ++i) p *= dims[i];
  return p;
}

template <typename T>
static TfLiteStatus TileEvalImpl(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* multiples =
      tflite::micro::GetEvalInput(context, node, kMultiplesTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  const T* input_data = tflite::micro::GetTensorData<T>(input);
  const int32_t* multiples_data = tflite::micro::GetTensorData<int32_t>(multiples);
  T* output_data = tflite::micro::GetTensorData<T>(output);

  const tflite::RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
  const tflite::RuntimeShape output_shape = tflite::micro::GetTensorShape(output);

  const int num_dims = input_shape.DimensionsCount();
  TF_LITE_ENSURE(context, num_dims > 0);
  TF_LITE_ENSURE(context, num_dims <= 8);
  TF_LITE_ENSURE_EQ(context, output_shape.DimensionsCount(), num_dims);

  int32_t input_dims[8];
  int32_t output_dims[8];
  int32_t input_strides[8];
  int32_t output_strides[8];

  for (int i = 0; i < num_dims; ++i) {
    input_dims[i] = input_shape.Dims(i);
    output_dims[i] = output_shape.Dims(i);
  }

  // Compute strides (row-major)
  int32_t stride = 1;
  for (int i = num_dims - 1; i >= 0; --i) {
    input_strides[i] = stride;
    stride *= input_dims[i];
  }
  stride = 1;
  for (int i = num_dims - 1; i >= 0; --i) {
    output_strides[i] = stride;
    stride *= output_dims[i];
  }

  const int32_t output_size = Product(output_dims, num_dims);

  // Map each output index to an input index by modulo per-axis.
  for (int32_t out_idx = 0; out_idx < output_size; ++out_idx) {
    int32_t in_idx = 0;
    int32_t t = out_idx;
    for (int axis = 0; axis < num_dims; ++axis) {
      const int32_t coord = t / output_strides[axis];
      t %= output_strides[axis];
      const int32_t in_coord = coord % input_dims[axis];
      in_idx += in_coord * input_strides[axis];
    }
    output_data[out_idx] = input_data[in_idx];
  }

  (void)multiples_data;  // multiples is implied by output shape.
  return kTfLiteOk;
}

// IMPORTANT: Use TfLiteEvalTensor in Prepare() to avoid allocating temporary
// TfLiteTensor wrappers (which must be manually freed in TFLM).
static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* multiples =
      tflite::micro::GetEvalInput(context, node, kMultiplesTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_TYPES_EQ(context, multiples->type, kTfLiteInt32);

  const tflite::RuntimeShape in_shape = tflite::micro::GetTensorShape(input);
  const tflite::RuntimeShape mult_shape = tflite::micro::GetTensorShape(multiples);

  // multiples is a 1-D tensor length == num_dims
  TF_LITE_ENSURE_EQ(context, mult_shape.DimensionsCount(), 1);
  TF_LITE_ENSURE_EQ(context, mult_shape.Dims(0), in_shape.DimensionsCount());

  return kTfLiteOk;
}

static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);

  switch (input->type) {
    case kTfLiteFloat32:
      return TileEvalImpl<float>(context, node);
    default:
      MicroPrintf("CUSTOM_TILE: unsupported type %d", input->type);
      return kTfLiteError;
  }
}

}  // namespace

TFLMRegistration Register_CUSTOM_TILE() {
  // builtin_code helps debugging; custom_name is unused for builtin ops.
  TFLMRegistration r{};
  r.init = nullptr;
  r.free = nullptr;
  r.prepare = Prepare;
  r.invoke = Invoke;
  r.reset = nullptr;
  r.builtin_code = static_cast<int32_t>(tflite::BuiltinOperator_TILE);
  r.custom_name = nullptr;
  return r;
}

}  // namespace custom
