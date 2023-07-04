#pragma once
#include "slang/macros.h"

BEGIN_SPEC(space_to_depth)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(BlockSize, REQUIRED)
DECLARE_SCALAR_PARAM(Layout, REQUIRED)

DECLARE_SIGNATURE(Input, Output, BlockSize, Layout)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(space_to_depth)