#pragma once
#include "slang/macros.h"

BEGIN_SPEC(prelu)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Alpha, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)


DECLARE_SIGNATURE(Input, Alpha, Output)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(prelu)