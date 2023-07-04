#pragma once
#include "slang/macros.h"

BEGIN_SPEC(concatenation)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Axis, REQUIRED)
DECLARE_SCALAR_PARAM(Input_cnt, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Axis, Input_cnt)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(concatenation)