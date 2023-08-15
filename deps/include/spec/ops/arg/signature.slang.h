START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i32_i32,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i8asym_i8asym,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, INT32, NO_QUANT),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(u8asym_u8asym,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, INT32, NO_QUANT),
                 SCALAR(Axis, INT32))


ADD_SIGNATURE(f32_f32,
              i32_i32,
              f16_f16,
              i8asym_i8asym,
              u8asym_u8asym)