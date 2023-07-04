#pragma once
#include "slang/macros.h"

BEGIN_SPEC(tile)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Multiples, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Multiples)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(tile)