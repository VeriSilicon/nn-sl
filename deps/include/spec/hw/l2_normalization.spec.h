#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/l2_normalization/spec.h"

OP_SIGNATURE_BEGIN(l2_normalization)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::l2_normalization::f32_f32),
    SELECT_SIGNATURE(true, op::l2_normalization::f16_f16),
    SELECT_SIGNATURE(true, op::l2_normalization::u8asymm_u8asymm),
    SELECT_SIGNATURE(true, op::l2_normalization::i8asymm_i8asymm),
    SELECT_SIGNATURE(true, op::l2_normalization::i8symm_i8symm)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::l2_normalization::rule_scale))

OP_SIGNATURE_END(l2_normalization)
