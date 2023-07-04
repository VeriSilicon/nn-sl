#pragma once
#include "slang/macros.h"

BEGIN_SPEC(gather)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Indices, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)
DECLARE_SCALAR_PARAM(Axis, REQUIRED)

DECLARE_SIGNATURE(Input, Indices, Output, Axis)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(gather)