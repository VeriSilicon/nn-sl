#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/reduce_max_min_prod_sum/spec.h"

OP_SIGNATURE_BEGIN(reduce_max_min_prod_sum)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::reduce_max_min_prod_sum::f32_f32),
    SELECT_SIGNATURE(true, op::reduce_max_min_prod_sum::f16_f16),
    SELECT_SIGNATURE(true, op::reduce_max_min_prod_sum::i32_i32),
    SELECT_SIGNATURE(true, op::reduce_max_min_prod_sum::i32_i16),
    SELECT_SIGNATURE(true, op::reduce_max_min_prod_sum::i16_i32),
    SELECT_SIGNATURE(true, op::reduce_max_min_prod_sum::u8asym_f16),
    SELECT_SIGNATURE(true, op::reduce_max_min_prod_sum::u8asym_u8asym),
    SELECT_SIGNATURE(true, op::reduce_max_min_prod_sum::i8dfp_i8dfp)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::reduce_max_min_prod_sum::rule_scale))

OP_SIGNATURE_END(reduce_max_min_prod_sum)
