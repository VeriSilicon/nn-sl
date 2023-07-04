#pragma once
#include "slang/macros.h"

BEGIN_SPEC(expand_dims)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Axis, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Axis)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(expand_dims)