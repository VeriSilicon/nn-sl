#pragma once
#include "slang/macros.h"

BEGIN_SPEC(svdf)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(WeightsFeature, REQUIRED)
DECLARE_TENSOR_PARAM(WeightsTime, REQUIRED)
DECLARE_TENSOR_PARAM(Bias, OPTIONAL)
DECLARE_TENSOR_PARAM(StateIn, REQUIRED)
DECLARE_TENSOR_PARAM(StateOut, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Rank, REQUIRED)
DECLARE_SCALAR_PARAM(NumUnits, REQUIRED)
DECLARE_SCALAR_PARAM(Activation, REQUIRED)

DECLARE_SIGNATURE(Input, WeightsFeature, WeightsTime, Bias, StateIn, StateOut, Output, Rank, NumUnits, Activation)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(svdf)