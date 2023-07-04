#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/mean/spec.h"

OP_SIGNATURE_BEGIN(mean)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::mean::f32_f32),
    SELECT_SIGNATURE(true, op::mean::f16_f16),
    SELECT_SIGNATURE(true, op::mean::u8asymm_u8asymm),
    SELECT_SIGNATURE(true, op::mean::i8asymm_i8asymm),
    SELECT_SIGNATURE(true, op::mean::i8symm_i8symm)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::mean::rule_scale))

OP_SIGNATURE_END(mean)
