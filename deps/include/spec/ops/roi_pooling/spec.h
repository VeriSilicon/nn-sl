#pragma once
#include "slang/macros.h"

BEGIN_SPEC(roi_pooling)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Regions, REQUIRED)
DECLARE_TENSOR_PARAM(BatchIndex, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(OutputHeight, REQUIRED)
DECLARE_SCALAR_PARAM(OutputWidth, REQUIRED)
DECLARE_SCALAR_PARAM(Scale, REQUIRED)
DECLARE_SCALAR_PARAM(Layout, REQUIRED)

DECLARE_SIGNATURE(Input, Regions, BatchIndex, Output, OutputHeight, OutputWidth, Scale, Layout)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(roi_pooling)