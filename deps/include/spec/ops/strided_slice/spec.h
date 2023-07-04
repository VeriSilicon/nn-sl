#pragma once
#include "slang/macros.h"

BEGIN_SPEC(strided_slice)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Begin, REQUIRED)
DECLARE_SCALAR_PARAM(End, REQUIRED)
DECLARE_SCALAR_PARAM(Strides, REQUIRED)
DECLARE_SCALAR_PARAM(Begin_mask, REQUIRED)
DECLARE_SCALAR_PARAM(End_mask, REQUIRED)
DECLARE_SCALAR_PARAM(Shrink_mask, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Begin, End, Strides, Begin_mask, End_mask, Shrink_mask)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(strided_slice)