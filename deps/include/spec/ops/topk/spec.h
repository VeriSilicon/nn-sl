#pragma once
#include "slang/macros.h"

BEGIN_SPEC(topk)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)
DECLARE_TENSOR_PARAM(Indices, REQUIRED)

DECLARE_SCALAR_PARAM(K, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Indices, K)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(topk)