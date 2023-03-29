#pragma once
#include "slang/macros.h"

BEGIN_SPEC(space_to_batch)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Block_size, REQUIRED)
DECLARE_SCALAR_PARAM(Pad, REQUIRED)
DECLARE_SCALAR_PARAM(Layout, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Block_size, Pad, Layout)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(space_to_batch)