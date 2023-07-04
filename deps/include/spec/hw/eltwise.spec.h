#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/eltwise/spec.h"

OP_SIGNATURE_BEGIN(eltwise)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::eltwise::f32_f32_f32),
    SELECT_SIGNATURE(true, op::eltwise::i32_i32_i32),
    SELECT_SIGNATURE(true, op::eltwise::i8asymm_i8asymm_i8asymm),
    SELECT_SIGNATURE(true, op::eltwise::i8symm_i8symm_i8symm),
    SELECT_SIGNATURE(true, op::eltwise::u8asymm_u8asymm_u8asymm),
    SELECT_SIGNATURE(true, op::eltwise::f32_f32_f32_const),
    SELECT_SIGNATURE(true, op::eltwise::f16_f16_f16_const),
    SELECT_SIGNATURE(true, op::eltwise::i32_i32_i32_const),
    SELECT_SIGNATURE(true, op::eltwise::i8asymm_i8asymm_i8asymm_const),
    SELECT_SIGNATURE(true, op::eltwise::i8symm_i8symm_i8symm_const),
    SELECT_SIGNATURE(true, op::eltwise::u8asymm_u8asymm_u8asymm_const)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::eltwise::rule_scale))

OP_SIGNATURE_END(eltwise)