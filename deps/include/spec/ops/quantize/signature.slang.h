START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f16_i8asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(f16_i8symm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT8, SYMM))

DEFINE_SIGNATURE(f16_u8asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(f16_u8symm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, UINT8, SYMM))

DEFINE_SIGNATURE(f32_u8asymm,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(f32_u8symm,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, UINT8, SYMM))

DEFINE_SIGNATURE(f32_i8asymm,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(f32_i8symm,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, INT8, SYMM))

DEFINE_SIGNATURE(i8asymm_i8asymm,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(i8symm_i8symm,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM))

DEFINE_SIGNATURE(i8asymm_u8asymm,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asymm_u8asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asymm_i8asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(u4asymm_u8asymm,
                 TENSOR(Input, UINT4, ASYMM),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u4symm_u8asymm,
                 TENSOR(Input, UINT4, SYMM),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asymm_u4asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, UINT4, ASYMM))

DEFINE_SIGNATURE(u8asymm_u4symm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, UINT4, SYMM))

DEFINE_SIGNATURE(i4asymm_u8asymm,
                 TENSOR(Input, INT4, ASYMM),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(i4symm_u8asymm,
                 TENSOR(Input, INT4, SYMM),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asymm_i4asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, INT4, ASYMM))

DEFINE_SIGNATURE(u8asymm_i4symm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, INT4, SYMM))

ADD_SIGNATURE(f16_i8asymm,
              f16_i8symm,
              f16_u8asymm,
              f16_u8symm,
              f32_u8asymm,
              f32_u8symm,
              f32_i8asymm,
              f32_i8symm,
              i8asymm_i8asymm,
              i8symm_i8symm,
              i8asymm_u8asymm,
              u8asymm_u8asymm,
              u8asymm_i8asymm,
              u4asymm_u8asymm,
              u4symm_u8asymm,
              u8asymm_u4asymm,
              u8asymm_u4symm,
              i4asymm_u8asymm,
              i4symm_u8asymm,
              u8asymm_i4asymm,
              u8asymm_i4symm)