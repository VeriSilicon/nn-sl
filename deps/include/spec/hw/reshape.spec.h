#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/reshape/spec.h"

OP_SIGNATURE_BEGIN(reshape)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::reshape::f32_f32),
    SELECT_SIGNATURE(true, op::reshape::f16_f16),
    SELECT_SIGNATURE(true, op::reshape::f16_f32),
    SELECT_SIGNATURE(true, op::reshape::f16_i32),
    SELECT_SIGNATURE(true, op::reshape::f16_u32),
    SELECT_SIGNATURE(true, op::reshape::f16_bf16),
    SELECT_SIGNATURE(true, op::reshape::f16_i16dfp),
    SELECT_SIGNATURE(true, op::reshape::f16_i16asymm),
    SELECT_SIGNATURE(true, op::reshape::f16_i16symm),
    SELECT_SIGNATURE(true, op::reshape::f16_i8dfp),
    SELECT_SIGNATURE(true, op::reshape::f16_i8asymm),
    SELECT_SIGNATURE(true, op::reshape::f16_i8symm),
    SELECT_SIGNATURE(true, op::reshape::f16_u8asymm),
    SELECT_SIGNATURE(true, op::reshape::f32_i32dfp),
    SELECT_SIGNATURE(true, op::reshape::f32_i32asymm),
    SELECT_SIGNATURE(true, op::reshape::f32_u32),
    SELECT_SIGNATURE(true, op::reshape::f32_f16),
    SELECT_SIGNATURE(true, op::reshape::f32_bf16),
    SELECT_SIGNATURE(true, op::reshape::f32_i16dfp),
    SELECT_SIGNATURE(true, op::reshape::f32_i8dfp),
    SELECT_SIGNATURE(true, op::reshape::f32_u8asymm),
    SELECT_SIGNATURE(true, op::reshape::i16dfp_f32),
    SELECT_SIGNATURE(true, op::reshape::i16dfp_i32),
    SELECT_SIGNATURE(true, op::reshape::i16dfp_u32),
    SELECT_SIGNATURE(true, op::reshape::i16dfp_i16dfp),
    SELECT_SIGNATURE(true, op::reshape::i16asymm_i16asymm),
    SELECT_SIGNATURE(true, op::reshape::i16symm_i16symm),
    SELECT_SIGNATURE(true, op::reshape::i16dfp_i8dfp),
    SELECT_SIGNATURE(true, op::reshape::i16dfp_u8asymm),
    SELECT_SIGNATURE(true, op::reshape::i16dfp_f16),
    SELECT_SIGNATURE(true, op::reshape::i16asymm_f16),
    SELECT_SIGNATURE(true, op::reshape::i16symm_f16),
    SELECT_SIGNATURE(true, op::reshape::i8dfp_f32),
    SELECT_SIGNATURE(true, op::reshape::i8symm_f16),
    SELECT_SIGNATURE(true, op::reshape::i8asymm_f16),
    SELECT_SIGNATURE(true, op::reshape::i8dfp_f16),
    SELECT_SIGNATURE(true, op::reshape::i8dfp_i32dfp),
    SELECT_SIGNATURE(true, op::reshape::i8dfp_u32),
    SELECT_SIGNATURE(true, op::reshape::i8dfp_i8dfp),
    SELECT_SIGNATURE(true, op::reshape::i8dfp_i8asymm),
    SELECT_SIGNATURE(true, op::reshape::i8dfp_i16dfp),
    SELECT_SIGNATURE(true, op::reshape::i8dfp_u8asymm),
    SELECT_SIGNATURE(true, op::reshape::i8asymm_i8asymm),
    SELECT_SIGNATURE(true, op::reshape::i8asymm_i8dfp),
    SELECT_SIGNATURE(true, op::reshape::i8symm_i8symm),
    SELECT_SIGNATURE(true, op::reshape::i8asymm_u8asymm),
    SELECT_SIGNATURE(true, op::reshape::u8asymm_u8asymm),
    SELECT_SIGNATURE(true, op::reshape::u8asymm_i8asymm),
    SELECT_SIGNATURE(true, op::reshape::u8asymm_f16),
    SELECT_SIGNATURE(true, op::reshape::u8asymm_i8dfp),
    SELECT_SIGNATURE(true, op::reshape::u8asymm_i16dfp),
    SELECT_SIGNATURE(true, op::reshape::u8asymm_i32asymm),
    SELECT_SIGNATURE(true, op::reshape::u8asymm_u32),
    SELECT_SIGNATURE(true, op::reshape::u8asymm_f32),
    SELECT_SIGNATURE(true, op::reshape::bf16_bf16),
    SELECT_SIGNATURE(true, op::reshape::bf16_f16),
    SELECT_SIGNATURE(true, op::reshape::bf16_f32),
    SELECT_SIGNATURE(true, op::reshape::i32_i32),
    SELECT_SIGNATURE(true, op::reshape::i32_i16dfp),
    SELECT_SIGNATURE(true, op::reshape::i32_i8dfp),
    SELECT_SIGNATURE(true, op::reshape::i32_u32),
    SELECT_SIGNATURE(true, op::reshape::i32_u16),
    SELECT_SIGNATURE(true, op::reshape::i32_u8asymm),
    SELECT_SIGNATURE(true, op::reshape::u32_u32),
    SELECT_SIGNATURE(true, op::reshape::u32_i16dfp),
    SELECT_SIGNATURE(true, op::reshape::u32_i8dfp),
    SELECT_SIGNATURE(true, op::reshape::u32_i32),
    SELECT_SIGNATURE(true, op::reshape::u32_u8asymm),
    SELECT_SIGNATURE(true, op::reshape::u32_u8),
    SELECT_SIGNATURE(true, op::reshape::bf16_i32),
    SELECT_SIGNATURE(true, op::reshape::i32_bf16),
    SELECT_SIGNATURE(true, op::reshape::u4asymm_u8asymm),
    SELECT_SIGNATURE(true, op::reshape::u4symm_u8asymm),
    SELECT_SIGNATURE(true, op::reshape::u8asymm_u4asymm),
    SELECT_SIGNATURE(true, op::reshape::u8asymm_u4symm),
    SELECT_SIGNATURE(true, op::reshape::i4asymm_u8asymm),
    SELECT_SIGNATURE(true, op::reshape::i4symm_u8asymm),
    SELECT_SIGNATURE(true, op::reshape::u8asymm_i4asymm),
    SELECT_SIGNATURE(true, op::reshape::u8asymm_i4symm)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::reshape::rule_scale))

OP_SIGNATURE_END(reshape)