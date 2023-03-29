#pragma once
#include "slang/macros.h"

BEGIN_SPEC(eltwise)

DECLARE_TENSOR_PARAM(Input0, REQUIRED)
DECLARE_TENSOR_PARAM(Input1, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Activation, OPTIONAL)

DECLARE_SIGNATURE(Input0, Input1, Output, Activation)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(eltwise)