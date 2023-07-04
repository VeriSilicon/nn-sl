#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/floor/spec.h"

OP_SIGNATURE_BEGIN(floor)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::floor::f32_f32),
    SELECT_SIGNATURE(true, op::floor::f32_f16),
    SELECT_SIGNATURE(true, op::floor::f16_f32),
    SELECT_SIGNATURE(true, op::floor::f16_f16),
    SELECT_SIGNATURE(true, op::floor::f16_u8asym),
    SELECT_SIGNATURE(true, op::floor::f16_i16dfp),
    SELECT_SIGNATURE(true, op::floor::f16_i16asym),
    SELECT_SIGNATURE(true, op::floor::f16_i16sym),
    SELECT_SIGNATURE(true, op::floor::f16_i8dfp),
    SELECT_SIGNATURE(true, op::floor::f16_i8asym),
    SELECT_SIGNATURE(true, op::floor::f16_i8sym),
    SELECT_SIGNATURE(true, op::floor::u8asymm_u8asym),
    SELECT_SIGNATURE(true, op::floor::i16dfp_f16),
    SELECT_SIGNATURE(true, op::floor::u8asymm_f16),
    SELECT_SIGNATURE(true, op::floor::i16asym_f16),
    SELECT_SIGNATURE(true, op::floor::i16sym_f16),
    SELECT_SIGNATURE(true, op::floor::i16dfp_i16dfp),
    SELECT_SIGNATURE(true, op::floor::i16asym_i16asym),
    SELECT_SIGNATURE(true, op::floor::i16sym_i16sym),
    SELECT_SIGNATURE(true, op::floor::i8dfp_f16),
    SELECT_SIGNATURE(true, op::floor::i8asym_f16),
    SELECT_SIGNATURE(true, op::floor::i8sym_f16),
    SELECT_SIGNATURE(true, op::floor::i8dfp_i8dfp),
    SELECT_SIGNATURE(true, op::floor::i8asym_i8asym),
    SELECT_SIGNATURE(true, op::floor::i8sym_i8sym)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::floor::rule_scale))

OP_SIGNATURE_END(floor)
