#pragma once
#include "slang/macros.h"

BEGIN_SPEC(select)

DECLARE_TENSOR_PARAM(Choose, REQUIRED)
DECLARE_TENSOR_PARAM(Input1, REQUIRED)
DECLARE_TENSOR_PARAM(Input2, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SIGNATURE(Choose, Input1, Input2, Output)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(select)