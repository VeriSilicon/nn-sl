#pragma once
#include "slang/macros.h"

BEGIN_SPEC(transpose_conv2d)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Kernel, REQUIRED)
DECLARE_TENSOR_PARAM(Bias, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Stride, REQUIRED)
DECLARE_SCALAR_PARAM(OutputPadding, REQUIRED)
DECLARE_SCALAR_PARAM(PadType, REQUIRED)
DECLARE_SCALAR_PARAM(Pad, REQUIRED)
DECLARE_SCALAR_PARAM(OutputShape, REQUIRED)
DECLARE_SCALAR_PARAM(Activation, REQUIRED)
DECLARE_SCALAR_PARAM(Layout, REQUIRED)

DECLARE_SIGNATURE(
    Input, Kernel, Bias, Output, Stride, OutputPadding, PadType, Pad, OutputShape, Activation, Layout)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(transpose_conv2d)
