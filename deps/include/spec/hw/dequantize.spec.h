#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/dequantize/spec.h"

OP_SIGNATURE_BEGIN(dequantize)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::dequantize::u8asym_f32),
    SELECT_SIGNATURE(true, op::dequantize::u8sym_f32),
    SELECT_SIGNATURE(true, op::dequantize::i8asym_f32),
    SELECT_SIGNATURE(true, op::dequantize::u8asym_f16),
    SELECT_SIGNATURE(true, op::dequantize::u8sym_f16),
    SELECT_SIGNATURE(true, op::dequantize::i8sym_f16),
    SELECT_SIGNATURE(true, op::dequantize::i8asymm_i8asymm),
    SELECT_SIGNATURE(true, op::dequantize::i8symm_i8symm),
    SELECT_SIGNATURE(true, op::dequantize::u8asymm_i8asymm),
    SELECT_SIGNATURE(true, op::dequantize::u4asymm_u8asymm),
    SELECT_SIGNATURE(true, op::dequantize::u4symm_u8asymm),
    SELECT_SIGNATURE(true, op::dequantize::u8asymm_u4asymm),
    SELECT_SIGNATURE(true, op::dequantize::u8asymm_u4symm),
    SELECT_SIGNATURE(true, op::dequantize::u8asymm_i4asymm),
    SELECT_SIGNATURE(true, op::dequantize::i4asymm_u8asymm),
    SELECT_SIGNATURE(true, op::dequantize::u8asymm_i4symm)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::dequantize::rule_scale))

OP_SIGNATURE_END(dequantize)
