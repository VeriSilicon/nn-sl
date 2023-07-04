#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/quantize/spec.h"

OP_SIGNATURE_BEGIN(quantize)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::quantize::f16_i8asymm),
    SELECT_SIGNATURE(true, op::quantize::f16_i8symm),
    SELECT_SIGNATURE(true, op::quantize::f16_u8asymm),
    SELECT_SIGNATURE(true, op::quantize::f16_u8symm),
    SELECT_SIGNATURE(true, op::quantize::f32_u8asymm),
    SELECT_SIGNATURE(true, op::quantize::f32_u8symm),
    SELECT_SIGNATURE(true, op::quantize::f32_i8asymm),
    SELECT_SIGNATURE(true, op::quantize::f32_i8symm),
    SELECT_SIGNATURE(true, op::quantize::i8asymm_i8asymm),
    SELECT_SIGNATURE(true, op::quantize::i8symm_i8symm),
    SELECT_SIGNATURE(true, op::quantize::i8asymm_u8asymm),
    SELECT_SIGNATURE(true, op::quantize::u8asymm_u8asymm),
    SELECT_SIGNATURE(true, op::quantize::u4asymm_u8asymm),
    SELECT_SIGNATURE(true, op::quantize::u4symm_u8asymm),
    SELECT_SIGNATURE(true, op::quantize::u8asymm_u4asymm),
    SELECT_SIGNATURE(true, op::quantize::u8asymm_u4symm),
    SELECT_SIGNATURE(true, op::quantize::i4asymm_u8asymm),
    SELECT_SIGNATURE(true, op::quantize::i4symm_u8asymm),
    SELECT_SIGNATURE(true, op::quantize::u8asymm_i4asymm),
    SELECT_SIGNATURE(true, op::quantize::u8asymm_i4symm),
    SELECT_SIGNATURE(true, op::quantize::u8asymm_i8asymm)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::quantize::rule_scale))

OP_SIGNATURE_END(quantize)
