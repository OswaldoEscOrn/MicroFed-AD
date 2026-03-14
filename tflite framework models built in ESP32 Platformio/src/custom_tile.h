#ifndef CUSTOM_TILE_H
#define CUSTOM_TILE_H

#include "tensorflow/lite/micro/micro_common.h"  // TFLMRegistration

namespace custom {

// Replacement kernel implementation for builtin TILE.
//
// IMPORTANT:
// - In TFLite Micro, leaking temporary tensors during Prepare() will trigger:
//   "All temp buffers must be freed before calling ResetTempAllocations()".
// - This kernel avoids allocating temporary TfLiteTensor wrappers in Prepare()
//   by using TfLiteEvalTensor helpers.

TFLMRegistration Register_CUSTOM_TILE();

}  // namespace custom

#endif  // CUSTOM_TILE_H