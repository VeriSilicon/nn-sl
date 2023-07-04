#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/prelu/spec.h"

OP_SIGNATURE_BEGIN(prelu)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::prelu::f32_f32_f32),
    SELECT_SIGNATURE(true, op::prelu::f16_f16_f16),
    SELECT_SIGNATURE(true, op::prelu::f16_f16_u8asym),
    SELECT_SIGNATURE(true, op::prelu::f16_f16_i16dfp),
    SELECT_SIGNATURE(true, op::prelu::f16_f16_i8dfp),
    SELECT_SIGNATURE(true, op::prelu::u8asym_f16_u8asym),
    SELECT_SIGNATURE(true, op::prelu::u8asym_f16_f16),
    SELECT_SIGNATURE(true, op::prelu::i8dfp_f16_i8dfp),
    SELECT_SIGNATURE(true, op::prelu::i8dfp_f16_f16),
    SELECT_SIGNATURE(true, op::prelu::i16dfp_f16_i16dfp),
    SELECT_SIGNATURE(true, op::prelu::i16dfp_f16_f16),
    SELECT_SIGNATURE(true, op::prelu::bf16_f16_bf16),
    SELECT_SIGNATURE(true, op::prelu::bf16_bf16_bf16),
    SELECT_SIGNATURE(true, op::prelu::i32_i32_i32),
    SELECT_SIGNATURE(true, op::prelu::i8asymm_i8asymm_i8asymm),
    SELECT_SIGNATURE(true, op::prelu::i8symm_i8symm_i8symm),
    SELECT_SIGNATURE(true, op::prelu::u8asymm_u8asymm_u8asymm),
    SELECT_SIGNATURE(true, op::prelu::u8asymm_u8asymm_f16),
    SELECT_SIGNATURE(true, op::prelu::f32_f32_f32_const),
    SELECT_SIGNATURE(true, op::prelu::f16_f16_f16_const),
    SELECT_SIGNATURE(true, op::prelu::f16_f16_u8asym_const),
    SELECT_SIGNATURE(true, op::prelu::f16_f16_i16dfp_const),
    SELECT_SIGNATURE(true, op::prelu::f16_f16_i8dfp_const),
    SELECT_SIGNATURE(true, op::prelu::u8asym_f16_u8asym_const),
    SELECT_SIGNATURE(true, op::prelu::u8asym_f16_f16_const),
    SELECT_SIGNATURE(true, op::prelu::i8dfp_f16_i8dfp_const),
    SELECT_SIGNATURE(true, op::prelu::i8dfp_f16_f16_const),
    SELECT_SIGNATURE(true, op::prelu::i16dfp_f16_i16dfp_const),
    SELECT_SIGNATURE(true, op::prelu::i16dfp_f16_f16_const),
    SELECT_SIGNATURE(true, op::prelu::bf16_f16_bf16_const),
    SELECT_SIGNATURE(true, op::prelu::bf16_bf16_bf16_const),
    SELECT_SIGNATURE(true, op::prelu::i32_i32_i32_const),
    SELECT_SIGNATURE(true, op::prelu::i8asymm_i8asymm_i8asymm_const),
    SELECT_SIGNATURE(true, op::prelu::i8symm_i8symm_i8symm_const),
    SELECT_SIGNATURE(true, op::prelu::u8asymm_u8asymm_u8asymm_const),
    SELECT_SIGNATURE(true, op::prelu::u8asymm_u8asymm_f16_const)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::prelu::rule_scale))

OP_SIGNATURE_END(prelu)
