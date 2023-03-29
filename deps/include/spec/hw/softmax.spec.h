#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/softmax/spec.h"

OP_SIGNATURE_BEGIN(softmax)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::softmax::f32_f32),
    SELECT_SIGNATURE(true, op::softmax::f32_f16),
    SELECT_SIGNATURE(true, op::softmax::f16_f16),
    SELECT_SIGNATURE(true, op::softmax::f16_f32),
    SELECT_SIGNATURE(true, op::softmax::f16_i16dfp),
    SELECT_SIGNATURE(true, op::softmax::f16_i16asymm),
    SELECT_SIGNATURE(true, op::softmax::f16_i16symm),
    SELECT_SIGNATURE(true, op::softmax::f16_i8dfp),
    SELECT_SIGNATURE(true, op::softmax::f16_i8symm),
    SELECT_SIGNATURE(true, op::softmax::f16_i8asymm),
    SELECT_SIGNATURE(true, op::softmax::f16_u8asymm),
    SELECT_SIGNATURE(true, op::softmax::bf16_bf16),
    SELECT_SIGNATURE(true, op::softmax::bf16_f32),
    SELECT_SIGNATURE(true, op::softmax::bf16_fp16),
    SELECT_SIGNATURE(true, op::softmax::u8asymm_u8asymm),
    SELECT_SIGNATURE(true, op::softmax::u8asymm_f16),
    SELECT_SIGNATURE(true, op::softmax::u8asymm_f32),
    SELECT_SIGNATURE(true, op::softmax::i8asymm_i8asymm),
    SELECT_SIGNATURE(true, op::softmax::i8symm_i8symm),
    SELECT_SIGNATURE(true, op::softmax::i8asymm_f16),
    SELECT_SIGNATURE(true, op::softmax::i8symm_f16),
    SELECT_SIGNATURE(true, op::softmax::i8asymm_f32),
    SELECT_SIGNATURE(true, op::softmax::i8symm_f32),
    SELECT_SIGNATURE(true, op::softmax::i8dfp_i8dfp),
    SELECT_SIGNATURE(true, op::softmax::i8dfp_f16),
    SELECT_SIGNATURE(true, op::softmax::i8dfp_f32),
    SELECT_SIGNATURE(true, op::softmax::i16asymm_i16asymm),
    SELECT_SIGNATURE(true, op::softmax::i16symm_i16symm),
    SELECT_SIGNATURE(true, op::softmax::i16asymm_f16),
    SELECT_SIGNATURE(true, op::softmax::i16symm_f16),
    SELECT_SIGNATURE(true, op::softmax::i16asymm_f32),
    SELECT_SIGNATURE(true, op::softmax::i16symm_f32),
    SELECT_SIGNATURE(true, op::softmax::i16dfp_f32),
    SELECT_SIGNATURE(true, op::softmax::i16dfp_f16),
    SELECT_SIGNATURE(true, op::softmax::i16dfp_i16dfp)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::softmax::rule_scale))

OP_SIGNATURE_END(softmax)
