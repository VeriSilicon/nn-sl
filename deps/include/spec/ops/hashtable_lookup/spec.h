#pragma once
#include "slang/macros.h"

BEGIN_SPEC(hashtable_lookup)

DECLARE_TENSOR_PARAM(Lookups, REQUIRED)
DECLARE_TENSOR_PARAM(Keys, REQUIRED)
DECLARE_TENSOR_PARAM(Values, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)
DECLARE_TENSOR_PARAM(Hits, REQUIRED)

DECLARE_SIGNATURE(Lookups, Keys, Values, Output, Hits)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(hashtable_lookup)