START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Indices, INT32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Indices, INT32, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(u8asym_u8asym,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Indices, INT32, NO_QUANT),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i8asym_i8asym,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Indices, INT32, NO_QUANT),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(f32_f32_const,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Indices, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(f16_f16_const,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Indices, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(u8asym_u8asym_const,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Indices, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i8asym_i8asym_const,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Indices, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(Axis, INT32))


ADD_SIGNATURE(f32_f32,
              f16_f16,
              u8asym_u8asym,
              i8asym_i8asym,
              f32_f32_const,
              f16_f16_const,
              u8asym_u8asym_const,
              i8asym_i8asym_const)