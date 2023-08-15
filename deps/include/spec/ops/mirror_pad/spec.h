#pragma once
#include "slang/macros.h"

BEGIN_SPEC(mirror_pad)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Pad, REQUIRED)
DECLARE_SCALAR_PARAM(PadMode, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Pad, PadMode)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(mirror_pad)