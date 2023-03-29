#pragma once
#include "slang/macros.h"

BEGIN_SPEC(reshape)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Shape, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SIGNATURE(Input, Shape, Output)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(reshape)