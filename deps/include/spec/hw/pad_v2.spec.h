#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/pad_v2/spec.h"

OP_SIGNATURE_BEGIN(pad_v2)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::pad_v2::f32_f32),
    SELECT_SIGNATURE(true, op::pad_v2::f32_bf16),
    SELECT_SIGNATURE(true, op::pad_v2::bf16_f32),
    SELECT_SIGNATURE(true, op::pad_v2::bf16_bf16),
    SELECT_SIGNATURE(true, op::pad_v2::f16_f16),
    SELECT_SIGNATURE(true, op::pad_v2::u8asymm_u8asymm),
    SELECT_SIGNATURE(true, op::pad_v2::i16dfp_i16dfp),
    SELECT_SIGNATURE(true, op::pad_v2::i16asymm_i16asymm),
    SELECT_SIGNATURE(true, op::pad_v2::i8symm_i8symm),
    SELECT_SIGNATURE(true, op::pad_v2::i32_i32),
    SELECT_SIGNATURE(true, op::pad_v2::u4asymm_u4asymm),
    SELECT_SIGNATURE(true, op::pad_v2::u4symm_u4symm),
    SELECT_SIGNATURE(true, op::pad_v2::i4asymm_i4asymm),
    SELECT_SIGNATURE(true, op::pad_v2::i4symm_i4symm)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::pad_v2::rule_scale))

OP_SIGNATURE_END(pad_v2)
