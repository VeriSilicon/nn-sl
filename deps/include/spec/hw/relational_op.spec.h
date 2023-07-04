#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/relational_op/spec.h"

OP_SIGNATURE_BEGIN(relational_op)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::relational_op::f32_f32),
    SELECT_SIGNATURE(true, op::relational_op::f16_f16),
    SELECT_SIGNATURE(true, op::relational_op::i32_i32),
    SELECT_SIGNATURE(true, op::relational_op::f16_i16dfp),
    SELECT_SIGNATURE(true, op::relational_op::f16_i16asymm),
    SELECT_SIGNATURE(true, op::relational_op::f16_i16symm),
    SELECT_SIGNATURE(true, op::relational_op::f16_i8dfp),
    SELECT_SIGNATURE(true, op::relational_op::f16_i8asymm),
    SELECT_SIGNATURE(true, op::relational_op::f16_i8symm),
    SELECT_SIGNATURE(true, op::relational_op::f16_u8asymm),
    SELECT_SIGNATURE(true, op::relational_op::i16dfp_i16dfp),
    SELECT_SIGNATURE(true, op::relational_op::i16asym_i16asymm),
    SELECT_SIGNATURE(true, op::relational_op::i16sym_i16symm),
    SELECT_SIGNATURE(true, op::relational_op::i8asymm_i8asymm),
    SELECT_SIGNATURE(true, op::relational_op::i8symm_i8symm),
    SELECT_SIGNATURE(true, op::relational_op::i8asymm_f16),
    SELECT_SIGNATURE(true, op::relational_op::i8symm_f16),
    SELECT_SIGNATURE(true, op::relational_op::u8asymm_f16),
    SELECT_SIGNATURE(true, op::relational_op::u8asymm_u8asym),
    SELECT_SIGNATURE(true, op::relational_op::bf16_bf16),
    SELECT_SIGNATURE(true, op::relational_op::bool8_bool8)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::relational_op::rule_scale))

OP_SIGNATURE_END(relational_op)
