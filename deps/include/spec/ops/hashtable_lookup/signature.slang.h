START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Lookups, INT32, NO_QUANT),
                 TENSOR(Keys, INT32, NO_QUANT),
                 TENSOR(Values, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 TENSOR(Hits, UINT8, ASYMM))

DEFINE_SIGNATURE(i32_i32,
                 TENSOR(Lookups, INT32, NO_QUANT),
                 TENSOR(Keys, INT32, NO_QUANT),
                 TENSOR(Values, INT32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT),
                 TENSOR(Hits, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asymm_u8asymm,
                 TENSOR(Lookups, INT32, NO_QUANT),
                 TENSOR(Keys, INT32, NO_QUANT),
                 TENSOR(Values, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM),
                 TENSOR(Hits, UINT8, ASYMM))

ADD_SIGNATURE(f32_f32,
              i32_i32,
              u8asymm_u8asymm)