START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 TENSOR(Axis, INT32, NO_QUANT),
                 SCALAR(Keep_dims, BOOL8))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 TENSOR(Axis, INT32, NO_QUANT),
                 SCALAR(Keep_dims, BOOL8))

DEFINE_SIGNATURE(u8asym_u8asym,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM),
                 TENSOR(Axis, INT32, NO_QUANT),
                 SCALAR(Keep_dims, BOOL8))

DEFINE_SIGNATURE(i8asym_i8asym,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM),
                 TENSOR(Axis, INT32, NO_QUANT),
                 SCALAR(Keep_dims, BOOL8))

ADD_SIGNATURE(f32_f32,
              f16_f16,
              u8asym_u8asym,
              i8asym_i8asym)