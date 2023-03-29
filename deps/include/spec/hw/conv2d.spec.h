#pragma once
#include "slang/tuple_filter.h"
#include "spec/hw/base.h"
#include "spec/ops/conv2d/spec.h"

OP_SIGNATURE_BEGIN(conv2d)

MAKE_SIGNATURE_TABLE(
    SELECT_SIGNATURE(T::NNCoreCount_FLOAT16 > 0 &&
                         (T::NN_POST_OUT_SUPPORT_FP16 || !T::NN_XYDP0),
                     op::conv2d::f16_f16_f16_f16),
    SELECT_SIGNATURE(T::NNCoreCount_FLOAT16 > 0 &&
                         (T::NN_POST_OUT_SUPPORT_FP16 || !T::NN_XYDP0),
                     op::conv2d::f16_f16_f32_f16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0,
                     op::conv2d::i8dfp_i8dfp_i32dfp_i8dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0,
                     op::conv2d::i8dfp_i8dfp_i32dfp_i8dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT16 > 0,
                     op::conv2d::i16dfp_i16dfp_i32dfp_i16dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT16 > 0,
                     op::conv2d::i16dfp_i16dfp_i64dfp_i16dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT16 > 0 && T::NN_XYDP0 &&
                         T::NN_POST_OUT_SUPPORT_FP16,
                     op::conv2d::i16dfp_i16dfp_i64dfp_f16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_u8asym_i32asym_u8asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION &&
                         (T::NN_POST_OUT_SUPPORT_FP16 || !T::NN_XYDP0),
                     op::conv2d::u8asym_u8asym_i32asym_f16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_i8asym_i32asym_u8asym),
    SELECT_SIGNATURE(T::NNCoreCount_BFLOAT && (!T::NN_XYDP0),
                     op::conv2d::bf16_bf16_f32_f32),
    SELECT_SIGNATURE(T::NNCoreCount_BFLOAT &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0),
                     op::conv2d::bf16_bf16_f32_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::i8asym_i8sympc_i32sympc_i8asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::i8sym_i8sym_i32sym_i8asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_i8sympc_i32sympc_u8asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_u8sympc_i32sympc_u8asym),
    SELECT_SIGNATURE(T::NNCoreCount_BFLOAT &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0),
                     op::conv2d::f32_bf16_f32_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_u8asym_i32asym_i8dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_u8asym_i32asym_i16dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0),
                     op::conv2d::u8asym_u8asym_i32asym_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION &&
                         !T::NN_XYDP0,
                     op::conv2d::u8asym_u8asym_i32asym_f32),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_u8sympc_i32sympc_i8dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_u8sympc_i32sympc_i16dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION &&
                         (T::NN_POST_OUT_SUPPORT_FP16 || !T::NN_XYDP0),
                     op::conv2d::u8asym_u8sympc_i32sympc_f16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0),
                     op::conv2d::u8asym_u8sympc_i32sympc_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION &&
                         !T::NN_XYDP0,
                     op::conv2d::u8asym_u8sympc_i32sympc_f32),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_i8dfp_i32dfp_u8asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_i8dfp_i32dfp_i8dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_i8dfp_i32dfp_i16dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION &&
                         (T::NN_POST_OUT_SUPPORT_FP16 || !T::NN_XYDP0),
                     op::conv2d::u8asym_i8dfp_i32dfp_f16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0),
                     op::conv2d::u8asym_i8dfp_i32dfp_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION &&
                         !T::NN_XYDP0,
                     op::conv2d::u8asym_i8dfp_i32dfp_f32),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_i8dfp_i64dfp_u8asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_i8dfp_i64dfp_i8dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_i8dfp_i64dfp_i16dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION &&
                         (T::NN_POST_OUT_SUPPORT_FP16 || !T::NN_XYDP0),
                     op::conv2d::u8asym_i8dfp_i64dfp_f16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0),
                     op::conv2d::u8asym_i8dfp_i64dfp_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION &&
                         !T::NN_XYDP0,
                     op::conv2d::u8asym_i8dfp_i64dfp_f32),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_i8sympc_i32sympc_u8asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_i8sympc_i32sympc_i8dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION,
                     op::conv2d::u8asym_i8sympc_i32sympc_i16dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION &&
                         (T::NN_POST_OUT_SUPPORT_FP16 || !T::NN_XYDP0),
                     op::conv2d::u8asym_i8sympc_i32sympc_f16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0),
                     op::conv2d::u8asym_i8sympc_i32sympc_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::TF_QUANTIZATION &&
                         !T::NN_XYDP0,
                     op::conv2d::u8asym_i8sympc_i32sympc_f32),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0,
                     op::conv2d::i8dfp_u8asym_i32asym_u8asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0,
                     op::conv2d::i8dfp_u8asym_i32asym_i8dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0,
                     op::conv2d::i8dfp_u8asym_i32asym_i16dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 &&
                         (T::NN_POST_OUT_SUPPORT_FP16 || !T::NN_XYDP0),
                     op::conv2d::i8dfp_u8asym_i32asym_f16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0),
                     op::conv2d::i8dfp_u8asym_i32asym_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && !T::NN_XYDP0,
                     op::conv2d::i8dfp_u8asym_i32asym_f32),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0,
                     op::conv2d::i8dfp_i8dfp_i32dfp_u8asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0,
                     op::conv2d::i8dfp_i8dfp_i32dfp_i16dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0),
                     op::conv2d::i8dfp_i8dfp_i32dfp_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && !T::NN_XYDP0,
                     op::conv2d::i8dfp_i8dfp_i32dfp_f32),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0,
                     op::conv2d::i8dfp_i8dfp_i64dfp_u8asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0,
                     op::conv2d::i8dfp_i8dfp_i64dfp_i16dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0),
                     op::conv2d::i8dfp_i8dfp_i64dfp_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && !T::NN_XYDP0,
                     op::conv2d::i8dfp_i8dfp_i64dfp_f32),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0,
                     op::conv2d::i8dfp_u8sympc_i32sympc_u8asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0,
                     op::conv2d::i8dfp_u8sympc_i32sympc_i8dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0,
                     op::conv2d::i8dfp_u8sympc_i32sympc_i16dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 &&
                         (T::NN_POST_OUT_SUPPORT_FP16 || !T::NN_XYDP0),
                     op::conv2d::i8dfp_u8sympc_i32sympc_f16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0),
                     op::conv2d::i8dfp_u8sympc_i32sympc_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && !T::NN_XYDP0,
                     op::conv2d::i8dfp_u8sympc_i32sympc_f32),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0,
                     op::conv2d::i8dfp_i8sympc_i32sympc_u8asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0,
                     op::conv2d::i8dfp_i8sympc_i32sympc_i16dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0),
                     op::conv2d::i8dfp_i8sympc_i32sympc_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && !T::NN_XYDP0,
                     op::conv2d::i8dfp_i8sympc_i32sympc_f32),
    SELECT_SIGNATURE(T::NNCoreCount_INT16 > 0,
                     op::conv2d::i16dfp_i16dfp_i32dfp_u8asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT16 > 0,
                     op::conv2d::i16dfp_i16dfp_i32dfp_i8dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT16 > 0 &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0),
                     op::conv2d::i16dfp_i16dfp_i32dfp_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_INT16 > 0 && !T::NN_XYDP0,
                     op::conv2d::i16dfp_i16dfp_i32dfp_f32),
    SELECT_SIGNATURE(T::NNCoreCount_INT16 > 0,
                     op::conv2d::i16dfp_i16dfp_i64dfp_u8asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT16 > 0,
                     op::conv2d::i16dfp_i16dfp_i64dfp_i8dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT16 > 0 &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0),
                     op::conv2d::i16dfp_i16dfp_i64dfp_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_INT16 > 0 && !T::NN_XYDP0,
                     op::conv2d::i16dfp_i16dfp_i64dfp_f32),
    SELECT_SIGNATURE(T::NNCoreCount_INT16 > 0,
                     op::conv2d::i16dfp_i16sympc_i64sympc_u8asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT16 > 0,
                     op::conv2d::i16dfp_i16sympc_i64sympc_i8dfp),
    SELECT_SIGNATURE(T::NNCoreCount_INT16 > 0 &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0),
                     op::conv2d::i16dfp_i16sympc_i64sympc_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_INT16 > 0 && !T::NN_XYDP0,
                     op::conv2d::i16dfp_i16sympc_i64sympc_f32),
    SELECT_SIGNATURE(T::NNCoreCount_FLOAT16 > 0 &&
                         T::NN_POST_OUT_SUPPORT_BF16 && T::NN_XYDP0 &&
                         T::NN_POST_MULT_SUPPORT_FP_CONV,
                     op::conv2d::f16_f16_f32_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_BFLOAT > 0 && T::NN_FLOAT32_IO &&
                         (T::NN_POST_OUT_SUPPORT_FP16 || !T::NN_XYDP0) &&
                         T::NN_POST_MULT_SUPPORT_FP_CONV,
                     op::conv2d::f32_bf16_f32_f16),
    SELECT_SIGNATURE(T::NNCoreCount_BFLOAT > 0 && T::NN_FLOAT32_IO &&
                         (T::NN_POST_OUT_SUPPORT_BF16 || !T::NN_XYDP0) &&
                         T::NN_POST_MULT_SUPPORT_FP_CONV,
                     op::conv2d::f32_bf16_f32_bf16),
    SELECT_SIGNATURE(T::NNCoreCount_BFLOAT > 0 && T::NN_FLOAT32_IO &&
                         T::NN_POST_MULT_SUPPORT_FP_CONV,
                     op::conv2d::f32_bf16_f32_f32),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::NN_4BIT_PHASE1 &&
                         T::TF_QUANTIZATION && T::NN_XYDP0,
                     op::conv2d::u4asym_u8asym_i32asym_u4asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::NN_4BIT_PHASE1 &&
                         T::TF_QUANTIZATION && T::NN_XYDP0,
                     op::conv2d::u4asym_u8asym_i32asym_i4asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::NN_4BIT_PHASE1 &&
                         T::TF_QUANTIZATION && T::NN_XYDP0,
                     op::conv2d::u4asym_i8asym_i32asym_u4asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::NN_4BIT_PHASE1 &&
                         T::TF_QUANTIZATION && T::NN_XYDP0,
                     op::conv2d::u4asym_i8asym_i32asym_i4asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::NN_4BIT_PHASE1 &&
                         T::TF_QUANTIZATION && T::NN_XYDP0,
                     op::conv2d::u4asym_i8sympc_i32asym_u4asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::NN_4BIT_PHASE1 &&
                         T::TF_QUANTIZATION && T::NN_XYDP0,
                     op::conv2d::u4asym_i8sympc_i32asym_i4asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::NN_4BIT_PHASE1 &&
                         T::NN_ASYMMETRIC_INT8 && T::NN_XYDP0,
                     op::conv2d::i4asym_i8asym_i32asym_u4asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::NN_4BIT_PHASE1 &&
                         T::TF_QUANTIZATION && T::NN_XYDP0,
                     op::conv2d::u4asym_i8asym_i32asym_i4asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::NN_4BIT_PHASE1 &&
                         T::NN_ASYMMETRIC_INT8 && T::NN_XYDP0,
                     op::conv2d::i4asym_u8asym_i32asym_u4asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::NN_4BIT_PHASE1 &&
                         T::TF_QUANTIZATION && T::NN_XYDP0,
                     op::conv2d::u4asym_u8asym_i32asym_i4asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::NN_4BIT_PHASE1 &&
                         T::NN_ASYMMETRIC_INT8 && T::NN_XYDP0,
                     op::conv2d::i4asym_i8sympc_i32sympc_u4asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::NN_4BIT_PHASE1 &&
                         T::NN_ASYMMETRIC_INT8 && T::NN_XYDP0,
                     op::conv2d::i4asym_i8sympc_i32sympc_i4asym),
    SELECT_SIGNATURE(T::NNCoreCount_INT8 > 0 && T::NN_4BIT_PHASE1 &&
                         T::NN_XYDP0,
                     op::conv2d::i4dfp_i8dfp_i32dfp_i4dfp))

MAKE_RULE_TABLE(SELECT_RULE(T::NNCoreCount == 4, op::conv2d::rule_scale),
                SELECT_RULE(T::EVIS_VX2, op::conv2d::rule_scale2))

OP_SIGNATURE_END(conv2d)
