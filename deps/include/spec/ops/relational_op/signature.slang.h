START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Input1, FP32, NO_QUANT),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(i32_i32,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Input1, INT32, NO_QUANT),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(f16_i16dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, INT16, NO_QUANT),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(f16_i16asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, INT16, ASYMM),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(f16_i16symm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, INT16, SYMM),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(f16_i8dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, INT16, DFP),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(f16_i8asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, INT8, ASYMM),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(f16_i8symm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, INT8, SYMM),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(f16_u8asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, UINT8, ASYMM),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(i16dfp_i16dfp,
                 TENSOR(Input, INT16, NO_QUANT),
                 TENSOR(Input1, INT16, NO_QUANT),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(i16asym_i16asymm,
                 TENSOR(Input, INT16, ASYMM),
                 TENSOR(Input1, INT16, ASYMM),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(i16sym_i16symm,
                 TENSOR(Input, INT16, SYMM),
                 TENSOR(Input1, INT16, SYMM),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(i8asymm_i8asymm,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Input1, INT8, ASYMM),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(i8symm_i8symm,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Input1, INT8, SYMM),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(i8asymm_f16,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(i8symm_f16,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(u8asymm_f16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(u8asymm_u8asym,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Input1, UINT8, ASYMM),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(bf16_bf16,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Input1, BF16, NO_QUANT),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(bool8_bool8,
                 TENSOR(Input, BOOL8, NO_QUANT),
                 TENSOR(Input1, BOOL8, NO_QUANT),
                 TENSOR(Output, BOOL8, NO_QUANT))

ADD_SIGNATURE(f32_f32,
              i32_i32,
              f16_f16,
              f16_i16dfp,
              f16_i16asymm,
              f16_i16symm,
              f16_i8dfp,
              f16_i8asymm,
              f16_i8symm,
              f16_u8asymm,
              i16dfp_i16dfp,
              i16asym_i16asymm,
              i16sym_i16symm,
              i8asymm_i8asymm,
              i8symm_i8symm,
              i8asymm_f16,
              i8symm_f16,
              u8asymm_f16,
              u8asymm_u8asym,
              bf16_bf16,
              bool8_bool8)