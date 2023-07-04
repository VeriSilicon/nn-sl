#pragma once
#include"slang/tuple_filter.h"
#include"spec/hw/base.h"
#include"spec/ops/space_to_batch/spec.h"

OP_SIGNATURE_BEGIN(space_to_batch)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(true, op::space_to_batch::f32_f32),
    SELECT_SIGNATURE(true, op::space_to_batch::f16_f16),
    SELECT_SIGNATURE(true, op::space_to_batch::u8asymm_u8asymm),
    SELECT_SIGNATURE(true, op::space_to_batch::i8asymm_i8asymm),
    SELECT_SIGNATURE(true, op::space_to_batch::i8symm_i8symm)
)

MAKE_RULE_TABLE(SELECT_RULE(true, op::space_to_batch::rule_scale))

OP_SIGNATURE_END(space_to_batch)
