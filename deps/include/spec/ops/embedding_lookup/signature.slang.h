START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32_f32,
                 TENSOR(Lookups, INT32, NO_QUANT),
                 TENSOR(Values, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_f16_f16,
                 TENSOR(Lookups, INT32, NO_QUANT),
                 TENSOR(Values, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i32_i32_i32,
                 TENSOR(Lookups, INT32, NO_QUANT),
                 TENSOR(Values, INT32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(i8asymm_i8asymm_i8asymm,
                 TENSOR(Lookups, INT32, NO_QUANT),
                 TENSOR(Values, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(i8symm_i8symm_i8symm,
                 TENSOR(Lookups, INT32, NO_QUANT),
                 TENSOR(Values, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM))

DEFINE_SIGNATURE(u8asymm_u8asymm_u8asymm,
                 TENSOR(Lookups, INT32, NO_QUANT),
                 TENSOR(Values, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM))

ADD_SIGNATURE(f32_f32_f32,
              f16_f16_f16,
              i32_i32_i32,
              i8asymm_i8asymm_i8asymm,
              i8symm_i8symm_i8symm,
              u8asymm_u8asymm_u8asymm)