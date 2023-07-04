#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/logical_not/spec.h"

OP_SIGNATURE_BEGIN(logical_not)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::logical_not::bool8_bool8)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::logical_not::rule_scale))

OP_SIGNATURE_END(logical_not)
