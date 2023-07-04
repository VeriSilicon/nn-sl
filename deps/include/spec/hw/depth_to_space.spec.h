#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/depth_to_space/spec.h"

OP_SIGNATURE_BEGIN(depth_to_space)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::depth_to_space::f32_f32),
    SELECT_SIGNATURE(true, op::depth_to_space::f16_f16),
    SELECT_SIGNATURE(true, op::depth_to_space::i8asym_i8asym),
    SELECT_SIGNATURE(true, op::depth_to_space::i8sym_i8sym),
    SELECT_SIGNATURE(true, op::depth_to_space::u8asym_u8asym)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::depth_to_space::rule_scale))

OP_SIGNATURE_END(depth_to_space)
