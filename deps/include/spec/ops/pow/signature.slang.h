START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32_f32,
                 TENSOR(Input0, FP32, NO_QUANT),
                 TENSOR(Input1, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_f16_f16,
                 TENSOR(Input0, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i32_i32_i32,
                 TENSOR(Input0, INT32, NO_QUANT),
                 TENSOR(Input1, INT32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(i8asymm_i8asymm_i8asymm,
                 TENSOR(Input0, INT8, ASYMM),
                 TENSOR(Input1, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(i8symm_i8symm_i8symm,
                 TENSOR(Input0, INT8, SYMM),
                 TENSOR(Input1, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM))

DEFINE_SIGNATURE(u8asymm_u8asymm_u8asymm,
                 TENSOR(Input0, UINT8, ASYMM),
                 TENSOR(Input1, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(f32_f32_f32_const,
                 TENSOR(Input0, FP32, NO_QUANT),
                 TENSOR(Input1, FP32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_f16_f16_const,
                 TENSOR(Input0, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i32_i32_i32_const,
                 TENSOR(Input0, INT32, NO_QUANT),
                 TENSOR(Input1, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(i8asymm_i8asymm_i8asymm_const,
                 TENSOR(Input0, INT8, ASYMM),
                 TENSOR(Input1, INT8, ASYMM, CONSTANT),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(i8symm_i8symm_i8symm_const,
                 TENSOR(Input0, INT8, SYMM),
                 TENSOR(Input1, INT8, SYMM, CONSTANT),
                 TENSOR(Output, INT8, SYMM))

DEFINE_SIGNATURE(u8asymm_u8asymm_u8asymm_const,
                 TENSOR(Input0, UINT8, ASYMM),
                 TENSOR(Input1, UINT8, ASYMM, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

ADD_SIGNATURE(f32_f32_f32,
              f16_f16_f16,
              i32_i32_i32,
              i8asymm_i8asymm_i8asymm,
              i8symm_i8symm_i8symm,
              u8asymm_u8asymm_u8asymm,
              f32_f32_f32_const,
              f16_f16_f16_const,
              i32_i32_i32_const,
              i8asymm_i8asymm_i8asymm_const,
              i8symm_i8symm_i8symm_const,
              u8asymm_u8asymm_u8asymm_const)