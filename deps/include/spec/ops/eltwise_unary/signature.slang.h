START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(i32_i32,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f32_f16,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(f32_bf16,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(f16_f32,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(f16_u8asym,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(f16_i8dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(f16_i8asym,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(f16_i8sym,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT8, SYMM))

DEFINE_SIGNATURE(f16_i16dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(bf16_bf16,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(bf16_f32,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(u8asym_u8asym,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asymm_f16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i8asym_i8asym,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(i8asym_f16,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, FP16, NO_QUANT))

/* HW 9.1.1 */
DEFINE_SIGNATURE(u4asym_u4asym,
                 TENSOR(Input, UINT4, ASYMM),
                 TENSOR(Output, UINT4, ASYMM))

DEFINE_SIGNATURE(u4sym_u4sym,
                 TENSOR(Input, UINT4, SYMM),
                 TENSOR(Output, UINT4, SYMM))

DEFINE_SIGNATURE(i4asym_i4asym,
                 TENSOR(Input, INT4, ASYMM),
                 TENSOR(Output, INT4, ASYMM))

DEFINE_SIGNATURE(i4sym_i4sym,
                 TENSOR(Input, INT4, SYMM),
                 TENSOR(Output, INT4, SYMM))

ADD_SIGNATURE(i32_i32,
              f32_f32,
              f32_f16,
              f32_bf16,
              f16_f32,
              f16_f16,
              f16_u8asym,
              f16_i8dfp,
              f16_i8asym,
              f16_i8sym,
              f16_i16dfp,
              bf16_bf16,
              bf16_f32,
              u8asym_u8asym,
              u8asymm_f16,
              i8asym_i8asym,
              i8asym_f16,
              u4asym_u4asym,
              u4sym_u4sym,
              i4asym_i4asym,
              i4sym_i4sym)