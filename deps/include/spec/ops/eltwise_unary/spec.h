#pragma once
#include "slang/macros.h"

BEGIN_SPEC(eltwise_unary)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SIGNATURE(Input, Output)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(eltwise_unary)