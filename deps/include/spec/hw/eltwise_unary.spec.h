#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/eltwise_unary/spec.h"

OP_SIGNATURE_BEGIN(eltwise_unary)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::eltwise_unary::i32_i32),
    SELECT_SIGNATURE(true, op::eltwise_unary::f32_f32),
    SELECT_SIGNATURE(true, op::eltwise_unary::f32_f16),
    SELECT_SIGNATURE(true, op::eltwise_unary::f32_bf16),
    SELECT_SIGNATURE(true, op::eltwise_unary::f16_f32),
    SELECT_SIGNATURE(true, op::eltwise_unary::f16_f16),
    SELECT_SIGNATURE(true, op::eltwise_unary::f16_u8asym),
    SELECT_SIGNATURE(true, op::eltwise_unary::f16_i8dfp),
    SELECT_SIGNATURE(true, op::eltwise_unary::f16_i8asym),
    SELECT_SIGNATURE(true, op::eltwise_unary::f16_i8sym),
    SELECT_SIGNATURE(true, op::eltwise_unary::f16_i16dfp),
    SELECT_SIGNATURE(true, op::eltwise_unary::bf16_bf16),
    SELECT_SIGNATURE(true, op::eltwise_unary::bf16_f32),
    SELECT_SIGNATURE(true, op::eltwise_unary::u8asym_u8asym),
    SELECT_SIGNATURE(true, op::eltwise_unary::u8asymm_f16),
    SELECT_SIGNATURE(true, op::eltwise_unary::i8asym_i8asym),
    SELECT_SIGNATURE(true, op::eltwise_unary::i8asym_f16),
    SELECT_SIGNATURE(true, op::eltwise_unary::u4asym_u4asym),
    SELECT_SIGNATURE(true, op::eltwise_unary::u4sym_u4sym),
    SELECT_SIGNATURE(true, op::eltwise_unary::i4asym_i4asym),
    SELECT_SIGNATURE(true, op::eltwise_unary::i4sym_i4sym)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::eltwise_unary::rule_scale))

OP_SIGNATURE_END(eltwise_unary)