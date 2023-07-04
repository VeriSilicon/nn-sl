#pragma once
#include "slang/macros.h"

BEGIN_SPEC(transpose)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Perm, REQUIRED)

DECLARE_SIGNATURE(
    Input, Output, Perm)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(transpose)
