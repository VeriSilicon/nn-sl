START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Choose, BOOL8, NO_QUANT),
                 TENSOR(Input1, FP32, NO_QUANT),
                 TENSOR(Input2, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Choose, BOOL8, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Input2, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i32_i32,
                 TENSOR(Choose, BOOL8, NO_QUANT),
                 TENSOR(Input1, INT32, NO_QUANT),
                 TENSOR(Input2, INT32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(u8asymm_u8asymm,
                 TENSOR(Choose, BOOL8, NO_QUANT),
                 TENSOR(Input1, UINT8, ASYMM),
                 TENSOR(Input2, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(i8asymm_i8asymm,
                 TENSOR(Choose, BOOL8, NO_QUANT),
                 TENSOR(Input1, INT8, ASYMM),
                 TENSOR(Input2, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM))

ADD_SIGNATURE(f32_f32,
              f16_f16,
              i32_i32,
              u8asymm_u8asymm,
              i8asymm_i8asymm)
