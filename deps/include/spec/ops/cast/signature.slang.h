START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(f16_f32,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_i32,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(f16_u8asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(f32_f16,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f32_i32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(f32_u8asymm,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(bool8_bool8,
                 TENSOR(Input, BOOL8, NO_QUANT),
                 TENSOR(Output, BOOL8, NO_QUANT))

DEFINE_SIGNATURE(u8asymm_u8asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(i8symm_i8symm,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM))

DEFINE_SIGNATURE(i32_i32,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(i32_f16,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i32_f32,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(i32_u8asymm,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(i32_i8symm,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Output, INT8, SYMM))

DEFINE_SIGNATURE(u16asymm_u16asymm,
                 TENSOR(Input, UINT16, ASYMM),
                 TENSOR(Output, UINT16, ASYMM))

DEFINE_SIGNATURE(i16symm_i16symm,
                 TENSOR(Input, INT16, SYMM),
                 TENSOR(Output, INT16, SYMM))

DEFINE_SIGNATURE(u8asymm_f16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(u8asymm_f32,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(u8asymm_i32,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, INT32, NO_QUANT))

ADD_SIGNATURE(f16_f16,
              f16_f32,
              f16_i32,
              f16_u8asymm,
              f32_f16,
              f32_f32,
              f32_i32,
              f32_u8asymm,
              bool8_bool8,
              u8asymm_u8asymm,
              i8symm_i8symm,
              i32_i32,
              i32_f16,
              i32_f32,
              i32_u8asymm,
              i32_i8symm,
              u16asymm_u16asymm,
              i16symm_i16symm,
              u8asymm_f16,
              u8asymm_f32,
              u8asymm_i32)