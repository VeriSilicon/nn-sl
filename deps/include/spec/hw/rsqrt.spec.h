#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/rsqrt/spec.h"

OP_SIGNATURE_BEGIN(rsqrt)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::rsqrt::f32_f32),
    SELECT_SIGNATURE(true, op::rsqrt::f16_f16),
    SELECT_SIGNATURE(true, op::rsqrt::f16_i16dfp),
    SELECT_SIGNATURE(true, op::rsqrt::f16_i16asym),
    SELECT_SIGNATURE(true, op::rsqrt::f16_i16sym),
    SELECT_SIGNATURE(true, op::rsqrt::f16_i8dfp),
    SELECT_SIGNATURE(true, op::rsqrt::f16_i8asym),
    SELECT_SIGNATURE(true, op::rsqrt::f16_i8sym),
    SELECT_SIGNATURE(true, op::rsqrt::f16_u8asym),
    SELECT_SIGNATURE(true, op::rsqrt::i8dfp_f16),
    SELECT_SIGNATURE(true, op::rsqrt::i8asym_f16),
    SELECT_SIGNATURE(true, op::rsqrt::i8sym_f16),
    SELECT_SIGNATURE(true, op::rsqrt::i8dfp_i8dfp),
    SELECT_SIGNATURE(true, op::rsqrt::i8asym_i8asym),
    SELECT_SIGNATURE(true, op::rsqrt::i8sym_i8sym),
    SELECT_SIGNATURE(true, op::rsqrt::u8asym_u8asym),
    SELECT_SIGNATURE(true, op::rsqrt::i16dfp_f16),
    SELECT_SIGNATURE(true, op::rsqrt::i16asym_f16),
    SELECT_SIGNATURE(true, op::rsqrt::i16sym_f16),
    SELECT_SIGNATURE(true, op::rsqrt::i16dfp_i16dfp),
    SELECT_SIGNATURE(true, op::rsqrt::i16asym_i16asym),
    SELECT_SIGNATURE(true, op::rsqrt::i16sym_i16sym),
    SELECT_SIGNATURE(true, op::rsqrt::bf16_bf16),
    SELECT_SIGNATURE(true, op::rsqrt::bf16_f32),
    SELECT_SIGNATURE(true, op::rsqrt::f32_bf16),
    SELECT_SIGNATURE(true, op::rsqrt::f32_u8asymm),
    SELECT_SIGNATURE(true, op::rsqrt::u8asymm_f32),
    SELECT_SIGNATURE(true, op::rsqrt::u8asymm_i16dfp),
    SELECT_SIGNATURE(true, op::rsqrt::u8asymm_i8dfp),
    SELECT_SIGNATURE(true, op::rsqrt::u8asymm_f16),
    SELECT_SIGNATURE(true, op::rsqrt::i8dfp_u8asym),
    SELECT_SIGNATURE(true, op::rsqrt::i8dfp_i16dfp),
    SELECT_SIGNATURE(true, op::rsqrt::i8dfp_bf16),
    SELECT_SIGNATURE(true, op::rsqrt::i8dfp_f32),
    SELECT_SIGNATURE(true, op::rsqrt::i16dfp_u8asym),
    SELECT_SIGNATURE(true, op::rsqrt::i16dfp_i8dfp),
    SELECT_SIGNATURE(true, op::rsqrt::i16dfp_bf16),
    SELECT_SIGNATURE(true, op::rsqrt::i16dfp_f32),
    SELECT_SIGNATURE(true, op::rsqrt::f16_bf16),
    SELECT_SIGNATURE(true, op::rsqrt::f16_f32),
    SELECT_SIGNATURE(true, op::rsqrt::u4asym_u4asym),
    SELECT_SIGNATURE(true, op::rsqrt::u4sym_u4sym),
    SELECT_SIGNATURE(true, op::rsqrt::i4asym_i4asym),
    SELECT_SIGNATURE(true, op::rsqrt::i4sym_i4sym)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::rsqrt::rule_scale))

OP_SIGNATURE_END(rsqrt)
