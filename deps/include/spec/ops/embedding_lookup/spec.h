#pragma once
#include "slang/macros.h"

BEGIN_SPEC(embedding_lookup)

DECLARE_TENSOR_PARAM(Lookups, REQUIRED)
DECLARE_TENSOR_PARAM(Values, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SIGNATURE(Lookups, Values, Output)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(embedding_lookup)