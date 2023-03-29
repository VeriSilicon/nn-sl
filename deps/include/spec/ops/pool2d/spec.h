#pragma once
#include "slang/macros.h"

BEGIN_SPEC(pool2d)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)
DECLARE_SCALAR_PARAM(Pad, OPTIONAL)
DECLARE_SCALAR_PARAM(PaddingCode, OPTIONAL)
DECLARE_SCALAR_PARAM(Stride, REQUIRED)
DECLARE_SCALAR_PARAM(Filter, REQUIRED)
DECLARE_SCALAR_PARAM(Activation, REQUIRED)
DECLARE_SCALAR_PARAM(Layout, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Pad, PaddingCode, Stride, Filter, Activation, Layout)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(pool2d)