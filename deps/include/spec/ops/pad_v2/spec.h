#pragma once
#include "slang/macros.h"

BEGIN_SPEC(pad_v2)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Pad, REQUIRED)
DECLARE_SCALAR_PARAM(Const_val, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Pad, Const_val)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(pad_v2)