#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/logical_and_or/spec.h"

OP_SIGNATURE_BEGIN(logical_and_or)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::logical_and_or::bool8_bool8)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::logical_and_or::rule_scale))

OP_SIGNATURE_END(logical_and_or)
