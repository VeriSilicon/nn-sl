#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/reduce_all_any/spec.h"

OP_SIGNATURE_BEGIN(reduce_all_any)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::reduce_all_any::bool8_bool8),
    SELECT_SIGNATURE(true, op::reduce_all_any::i8_i8)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::reduce_all_any::rule_scale))

OP_SIGNATURE_END(reduce_all_any)
