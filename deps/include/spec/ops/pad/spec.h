#pragma once
#include "slang/macros.h"

BEGIN_SPEC(pad)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Pad, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Pad)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(pad)