START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, FP32))

DEFINE_SIGNATURE(f32_bf16,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, BF16, NO_QUANT),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, FP32))

DEFINE_SIGNATURE(bf16_f32,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, BF16))

DEFINE_SIGNATURE(bf16_bf16,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Output, BF16, NO_QUANT),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, BF16))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, FP16))

DEFINE_SIGNATURE(u8asymm_u8asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, INT32))

DEFINE_SIGNATURE(i16dfp_i16dfp,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Output, INT16, DFP),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, INT32))

DEFINE_SIGNATURE(i16asymm_i16asymm,
                 TENSOR(Input, INT16, ASYMM),
                 TENSOR(Output, INT16, ASYMM),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, INT32))

DEFINE_SIGNATURE(i16symm_i16symm,
                 TENSOR(Input, INT16, SYMM),
                 TENSOR(Output, INT16, SYMM),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, INT32))

DEFINE_SIGNATURE(i8dfp_i8dfp,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Output, INT8, DFP),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, INT32))

DEFINE_SIGNATURE(i8asymm_i8asymm,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, INT32))

DEFINE_SIGNATURE(i8symm_i8symm,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, INT32))

DEFINE_SIGNATURE(i32_i32,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, INT32))

/* HW 9.1.1 */
DEFINE_SIGNATURE(u4asymm_u4asymm,
                 TENSOR(Input, UINT4, ASYMM),
                 TENSOR(Output, UINT4, ASYMM),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, INT32))

DEFINE_SIGNATURE(u4symm_u4symm,
                 TENSOR(Input, UINT4, SYMM),
                 TENSOR(Output, UINT4, SYMM),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, INT32))

DEFINE_SIGNATURE(i4asymm_i4asymm,
                 TENSOR(Input, INT4, ASYMM),
                 TENSOR(Output, INT4, ASYMM),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, INT32))

DEFINE_SIGNATURE(i4symm_i4symm,
                 TENSOR(Input, INT4, SYMM),
                 TENSOR(Output, INT4, SYMM),
                 SCALAR(Pad, INT32),
                 SCALAR(Const_val, INT32))

ADD_SIGNATURE(f32_f32,
              f32_bf16,
              bf16_f32,
              bf16_bf16,
              f16_f16,
              u8asymm_u8asymm,
              i16dfp_i16dfp,
              i16asymm_i16asymm,
              i16symm_i16symm,
              i8dfp_i8dfp,
              i8asymm_i8asymm,
              i8symm_i8symm,
              i32_i32,
              u4asymm_u4asymm,
              u4symm_u4symm,
              i4asymm_i4asymm,
              i4symm_i4symm)