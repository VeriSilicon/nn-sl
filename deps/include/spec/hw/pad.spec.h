#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/pad/spec.h"

OP_SIGNATURE_BEGIN(pad)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::pad::f32_f32),
    SELECT_SIGNATURE(true, op::pad::f32_bf16),
    SELECT_SIGNATURE(true, op::pad::bf16_f32),
    SELECT_SIGNATURE(true, op::pad::bf16_bf16),
    SELECT_SIGNATURE(true, op::pad::f16_f16),
    SELECT_SIGNATURE(true, op::pad::u8asymm_u8asymm),
    SELECT_SIGNATURE(true, op::pad::i16dfp_i16dfp),
    SELECT_SIGNATURE(true, op::pad::i16asymm_i16asymm),
    SELECT_SIGNATURE(true, op::pad::i8symm_i8symm),
    SELECT_SIGNATURE(true, op::pad::i32_i32),
    SELECT_SIGNATURE(true, op::pad::u4asymm_u4asymm),
    SELECT_SIGNATURE(true, op::pad::u4symm_u4symm),
    SELECT_SIGNATURE(true, op::pad::i4asymm_i4asymm),
    SELECT_SIGNATURE(true, op::pad::i4symm_i4symm)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::pad::rule_scale))

OP_SIGNATURE_END(pad)
