#pragma once
#include "slang/macros.h"

BEGIN_SPEC(channel_shuffle)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Groups, REQUIRED)
DECLARE_SCALAR_PARAM(Axis, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Groups, Axis)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(channel_shuffle)