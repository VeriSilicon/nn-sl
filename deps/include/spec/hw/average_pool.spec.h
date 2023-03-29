#pragma once
#include "slang/tuple_filter.h"
#include "spec/hw/base.h"
#include "spec/ops/average_pool2d/spec.h"

OP_SIGNATURE_BEGIN(average_pool2d)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::average_pool2d::f32_f32),
    SELECT_SIGNATURE(true, op::average_pool2d::f32_f16),
    SELECT_SIGNATURE(true, op::average_pool2d::f16_f32),
    SELECT_SIGNATURE(true, op::average_pool2d::f16_f16),
    SELECT_SIGNATURE(true, op::average_pool2d::f16_u8asymm),
    SELECT_SIGNATURE(true, op::average_pool2d::f16_i8dfp),
    SELECT_SIGNATURE(true, op::average_pool2d::f16_i8asymm),
    SELECT_SIGNATURE(true, op::average_pool2d::f16_i8symm),
    SELECT_SIGNATURE(true, op::average_pool2d::f16_i16dfp),
    SELECT_SIGNATURE(true, op::average_pool2d::f16_i16asymm),
    SELECT_SIGNATURE(true, op::average_pool2d::f16_i16symm),
    SELECT_SIGNATURE(true, op::average_pool2d::bf16_bf16),
    SELECT_SIGNATURE(true, op::average_pool2d::u8asymm_u8asymm),
    SELECT_SIGNATURE(true, op::average_pool2d::u8asymm_f16),
    SELECT_SIGNATURE(true, op::average_pool2d::i8asymm_i8asymm),
    SELECT_SIGNATURE(true, op::average_pool2d::i8symm_i8symm),
    SELECT_SIGNATURE(true, op::average_pool2d::i8asymm_f16),
    SELECT_SIGNATURE(true, op::average_pool2d::i8symm_f16),
    SELECT_SIGNATURE(true, op::average_pool2d::i8dfp_i8dfp),
    SELECT_SIGNATURE(true, op::average_pool2d::i8dfp_f16),
    SELECT_SIGNATURE(true, op::average_pool2d::i16dfp_i16dfp),
    SELECT_SIGNATURE(true, op::average_pool2d::i16asymm_i16asymm),
    SELECT_SIGNATURE(true, op::average_pool2d::i16symm_i16symm),
    SELECT_SIGNATURE(true, op::average_pool2d::i16dfp_f16),
    SELECT_SIGNATURE(true, op::average_pool2d::i16asymm_f16),
    SELECT_SIGNATURE(true, op::average_pool2d::i16symm_f16),
    SELECT_SIGNATURE(true, op::average_pool2d::u8asymm_i16dfp),
    SELECT_SIGNATURE(true, op::average_pool2d::u8asymm_bf16),
    SELECT_SIGNATURE(true, op::average_pool2d::u8asymm_f32),
    SELECT_SIGNATURE(true, op::average_pool2d::u8asymm_i8dfp),
    SELECT_SIGNATURE(true, op::average_pool2d::i8dfp_u8asymm),
    SELECT_SIGNATURE(true, op::average_pool2d::i8dfp_i16dfp),
    SELECT_SIGNATURE(true, op::average_pool2d::i8dfp_bf16),
    SELECT_SIGNATURE(true, op::average_pool2d::i8dfp_f32),
    SELECT_SIGNATURE(true, op::average_pool2d::i16dfp_u8asymm),
    SELECT_SIGNATURE(true, op::average_pool2d::i16dfp_i8dfp),
    SELECT_SIGNATURE(true, op::average_pool2d::i16dfp_bf16),
    SELECT_SIGNATURE(true, op::average_pool2d::i16dfp_f32),
    SELECT_SIGNATURE(true, op::average_pool2d::f32_bf16),
    SELECT_SIGNATURE(true, op::average_pool2d::bf16_f32),
    SELECT_SIGNATURE(true, op::average_pool2d::f16_bf16))

MAKE_RULE_TABLE(SELECT_RULE(true, op::average_pool2d::rule_scale))

OP_SIGNATURE_END(average_pool2d)
