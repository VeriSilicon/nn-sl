#pragma once
#include "slang/macros.h"

BEGIN_SPEC(batch_to_space)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Block_size, REQUIRED)
DECLARE_SCALAR_PARAM(Crop, REQUIRED)
DECLARE_SCALAR_PARAM(Layout, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Block_size, Crop, Layout)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(batch_to_space)