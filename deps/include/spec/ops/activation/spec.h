#pragma once
#include "slang/macros.h"

BEGIN_SPEC(activation)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)
DECLARE_SCALAR_PARAM(Alpha, OPTIONAL)

DECLARE_SIGNATURE(Input, Output, Alpha)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(activation)