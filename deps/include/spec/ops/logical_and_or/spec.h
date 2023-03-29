#pragma once
#include "slang/macros.h"

BEGIN_SPEC(logical_and_or)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Input1, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SIGNATURE(Input, Input1, Output)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(logical_and_or)