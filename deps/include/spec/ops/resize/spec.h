#pragma once
#include "slang/macros.h"

BEGIN_SPEC(resize)

DECLARE_TENSOR_PARAM(Input, REQUIRED)
DECLARE_TENSOR_PARAM(Output, REQUIRED)

DECLARE_SCALAR_PARAM(Output_width, REQUIRED)
DECLARE_SCALAR_PARAM(Output_height, REQUIRED)
DECLARE_SCALAR_PARAM(Factor, OPTIONAL)
DECLARE_SCALAR_PARAM(Layout, REQUIRED)
DECLARE_SCALAR_PARAM(Align_corners, REQUIRED)
DECLARE_SCALAR_PARAM(Half_pixel_centers, REQUIRED)

DECLARE_SIGNATURE(Input, Output, Output_width, Output_height, Factor, Layout, Align_corners, Half_pixel_centers)

#include "signature.slang.h"
#include "rule.slang.h"
END_SPEC(resize)