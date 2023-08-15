#pragma once
#include "slang/macros.h"

BEGIN_SPEC(fully_connected)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Weight, REQUIRED)
DECLARE_TENSOR_PARAM(Bias, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SIGNATURE(Input, Weight, Bias, Output)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(fully_connected)