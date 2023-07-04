START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Begin, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Size, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Begin, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Size, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i32_i32,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Begin, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Size, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(u8asymm_u8asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Begin, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Size, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(i8asymm_i8asymm,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Begin, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Size, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, ASYMM))

ADD_SIGNATURE(f32_f32,
              f16_f16,
              i32_i32,
              u8asymm_u8asymm,
              i8asymm_i8asymm)
