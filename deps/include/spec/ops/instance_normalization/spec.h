#pragma once
#include "slang/macros.h"

BEGIN_SPEC(instance_normalization)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Epsilon, REQUIRED)
DECLARE_SCALAR_PARAM(Layout, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Epsilon, Layout)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(instance_normalization)