#pragma once
#include "slang/macros.h"

BEGIN_SPEC(batch_matmul)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Input2, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)
DECLARE_SCALAR_PARAM(Adj_x, REQUIRED)
DECLARE_SCALAR_PARAM(Adj_y, REQUIRED)

DECLARE_SIGNATURE(Input, Input2, Output, Adj_x, Adj_y)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(batch_matmul)