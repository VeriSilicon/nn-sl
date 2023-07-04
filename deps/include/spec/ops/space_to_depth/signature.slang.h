START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(BlockSize, INT32),
                 SCALAR(Layout, BOOL8))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(BlockSize, INT32),
                 SCALAR(Layout, BOOL8))

DEFINE_SIGNATURE(u8asymm_u8asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(BlockSize, INT32),
                 SCALAR(Layout, BOOL8))

DEFINE_SIGNATURE(i8asymm_i8asymm,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(BlockSize, INT32),
                 SCALAR(Layout, BOOL8))

DEFINE_SIGNATURE(i8symm_i8symm,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM),
                 SCALAR(BlockSize, INT32),
                 SCALAR(Layout, BOOL8))

ADD_SIGNATURE(f32_f32,
              f16_f16,
              u8asymm_u8asymm,
              i8asymm_i8asymm,
              i8symm_i8symm)