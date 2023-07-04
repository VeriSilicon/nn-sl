#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/space_to_depth/spec.h"

OP_SIGNATURE_BEGIN(space_to_depth)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::space_to_depth::f32_f32),
    SELECT_SIGNATURE(true, op::space_to_depth::f16_f16),
    SELECT_SIGNATURE(true, op::space_to_depth::u8asymm_u8asymm),
    SELECT_SIGNATURE(true, op::space_to_depth::i8asymm_i8asymm),
    SELECT_SIGNATURE(true, op::space_to_depth::i8symm_i8symm)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::space_to_depth::rule_scale))

OP_SIGNATURE_END(space_to_depth)
