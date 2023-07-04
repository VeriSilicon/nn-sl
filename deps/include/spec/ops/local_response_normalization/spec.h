#pragma once
#include "slang/macros.h"

BEGIN_SPEC(local_response_normalization)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Radius, REQUIRED)
DECLARE_SCALAR_PARAM(Bias, REQUIRED)
DECLARE_SCALAR_PARAM(Alpha, REQUIRED)
DECLARE_SCALAR_PARAM(Beta, REQUIRED)
DECLARE_SCALAR_PARAM(Axis, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Radius, Bias, Alpha, Beta, Axis)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(local_response_normalization)