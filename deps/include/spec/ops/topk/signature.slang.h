START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 TENSOR(Indices, INT32, NO_QUANT),
                 SCALAR(K, INT32))

DEFINE_SIGNATURE(i32_i32,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT),
                 TENSOR(Indices, INT32, NO_QUANT),
                 SCALAR(K, INT32))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 TENSOR(Indices, INT32, NO_QUANT),
                 SCALAR(K, INT32))

DEFINE_SIGNATURE(u8asymm_u8asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM),
                 TENSOR(Indices, INT32, NO_QUANT),
                 SCALAR(K, INT32))

DEFINE_SIGNATURE(i8asymm_i8asymm,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM),
                 TENSOR(Indices, INT32, NO_QUANT),
                 SCALAR(K, INT32))

DEFINE_SIGNATURE(i8symm_i8symm,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM),
                 TENSOR(Indices, INT32, NO_QUANT),
                 SCALAR(K, INT32))

ADD_SIGNATURE(f32_f32,
              f16_f16,
              u8asymm_u8asymm,
              i8asymm_i8asymm,
              i8symm_i8symm)