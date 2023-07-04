#pragma once
#include "slang/macros.h"

BEGIN_SPEC(mean)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)
DECLARE_TENSOR_PARAM(Axis, REQUIRED)

DECLARE_SCALAR_PARAM(KeepDims, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Axis, KeepDims)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(mean)