START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Input1, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_f32,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_i32,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(f16_u32,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, UINT32, NO_QUANT))

DEFINE_SIGNATURE(f16_bf16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(f16_i16dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(f16_i16asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, INT16, ASYMM))

DEFINE_SIGNATURE(f16_i16symm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, INT16, SYMM))

DEFINE_SIGNATURE(f16_i8dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(f16_i8asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(f16_i8symm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, INT8, SYMM))

DEFINE_SIGNATURE(f16_u8asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(f32_i32dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP32, NO_QUANT),
                 TENSOR(Output, INT32, DFP))

DEFINE_SIGNATURE(f32_i32asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP32, NO_QUANT),
                 TENSOR(Output, INT32, ASYMM))

DEFINE_SIGNATURE(f32_u32,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP32, NO_QUANT),
                 TENSOR(Output, UINT32, NO_QUANT))

DEFINE_SIGNATURE(f32_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input1, FP32, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT))


ADD_SIGNATURE(f32_f32,
              f16_f16,
              f16_f32,
              f16_i32,
              f16_u32,
              f16_bf16,
              f16_i16dfp,
              f16_i16asymm,
              f16_i16asymm,
              f16_i16symm,
              f16_i8dfp,
              f16_i8asymm,
              f16_i8symm,
              f16_u8asymm,
              f32_i32dfp,
              f32_i32asymm,
              f32_u32,
              f32_f16)