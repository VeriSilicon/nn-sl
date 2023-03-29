#pragma once
#include "slang/macros.h"

BEGIN_SPEC(pow)

DECLARE_TENSOR_PARAM(Input0, REQUIRED)
DECLARE_TENSOR_PARAM(Input1, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)


DECLARE_SIGNATURE(Input0, Input1, Output)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(pow)