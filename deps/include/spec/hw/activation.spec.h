#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/activation/spec.h"

OP_SIGNATURE_BEGIN(activation)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::activation::f32_f32),
    SELECT_SIGNATURE(true, op::activation::f16_f16),
    SELECT_SIGNATURE(true, op::activation::f16_i16dfp),
    SELECT_SIGNATURE(true, op::activation::f16_i16asym),
    SELECT_SIGNATURE(true, op::activation::f16_i16sym),
    SELECT_SIGNATURE(true, op::activation::f16_i8dfp),
    SELECT_SIGNATURE(true, op::activation::f16_i8asym),
    SELECT_SIGNATURE(true, op::activation::f16_i8sym),
    SELECT_SIGNATURE(true, op::activation::f16_u8asym),
    SELECT_SIGNATURE(true, op::activation::i8dfp_f16),
    SELECT_SIGNATURE(true, op::activation::i8asym_f16),
    SELECT_SIGNATURE(true, op::activation::i8sym_f16),
    SELECT_SIGNATURE(true, op::activation::i8dfp_i8dfp),
    SELECT_SIGNATURE(true, op::activation::i8asym_i8asym),
    SELECT_SIGNATURE(true, op::activation::i8sym_i8sym),
    SELECT_SIGNATURE(true, op::activation::u8asym_u8asym),
    SELECT_SIGNATURE(true, op::activation::i16dfp_f16),
    SELECT_SIGNATURE(true, op::activation::i16asym_f16),
    SELECT_SIGNATURE(true, op::activation::i16sym_f16),
    SELECT_SIGNATURE(true, op::activation::i16dfp_i16dfp),
    SELECT_SIGNATURE(true, op::activation::i16asym_i16asym),
    SELECT_SIGNATURE(true, op::activation::i16sym_i16sym),
    SELECT_SIGNATURE(true, op::activation::bf16_bf16),
    SELECT_SIGNATURE(true, op::activation::bf16_f32),
    SELECT_SIGNATURE(true, op::activation::f32_bf16),
    SELECT_SIGNATURE(true, op::activation::f32_u8asymm),
    SELECT_SIGNATURE(true, op::activation::u8asymm_f32),
    SELECT_SIGNATURE(true, op::activation::u8asymm_i8dfp),
    SELECT_SIGNATURE(true, op::activation::u8asymm_i16dfp),
    SELECT_SIGNATURE(true, op::activation::u8asymm_i8dfp),
    SELECT_SIGNATURE(true, op::activation::u8asymm_f16),
    SELECT_SIGNATURE(true, op::activation::i8dfp_u8asym),
    SELECT_SIGNATURE(true, op::activation::i8dfp_i16dfp),
    SELECT_SIGNATURE(true, op::activation::i8dfp_bf16),
    SELECT_SIGNATURE(true, op::activation::i8dfp_f32),
    SELECT_SIGNATURE(true, op::activation::i16dfp_u8asym),
    SELECT_SIGNATURE(true, op::activation::i16dfp_i8dfp),
    SELECT_SIGNATURE(true, op::activation::i16dfp_bf16),
    SELECT_SIGNATURE(true, op::activation::i16dfp_f32),
    SELECT_SIGNATURE(true, op::activation::f16_bf16),
    SELECT_SIGNATURE(true, op::activation::f16_f32),
    SELECT_SIGNATURE(true, op::activation::u4asym_u4asym),
    SELECT_SIGNATURE(true, op::activation::u4sym_u4sym),
    SELECT_SIGNATURE(true, op::activation::i4asym_i4asym),
    SELECT_SIGNATURE(true, op::activation::i4sym_i4sym)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::activation::rule_scale))

OP_SIGNATURE_END(activation)
