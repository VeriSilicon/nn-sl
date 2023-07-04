#pragma once
#include "slang/macros.h"

BEGIN_SPEC(log_softmax)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Beta, REQUIRED)
DECLARE_SCALAR_PARAM(Axis, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Beta, Axis)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(log_softmax)