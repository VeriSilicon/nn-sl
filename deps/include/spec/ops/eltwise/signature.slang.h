START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32_f32,
                 TENSOR(Input0, FP32, NO_QUANT),
                 TENSOR(Input1, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(f16_f16_f16,
                 TENSOR(Input0, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(i32_i32_i32,
                 TENSOR(Input0, INT32, NO_QUANT),
                 TENSOR(Input1, INT32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(i8asymm_i8asymm_i8asymm,
                 TENSOR(Input0, INT8, ASYMM),
                 TENSOR(Input1, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(i8symm_i8symm_i8symm,
                 TENSOR(Input0, INT8, SYMM),
                 TENSOR(Input1, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(u8asymm_u8asymm_u8asymm,
                 TENSOR(Input0, UINT8, ASYMM),
                 TENSOR(Input1, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(f32_f32_f32_const1,
                 TENSOR(Input0, FP32, NO_QUANT, CONSTANT),
                 TENSOR(Input1, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(f16_f16_f16_const1,
                 TENSOR(Input0, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Input1, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(i32_i32_i32_const1,
                 TENSOR(Input0, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Input1, INT32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(i8asymm_i8asymm_i8asymm_const1,
                 TENSOR(Input0, INT8, ASYMM, CONSTANT),
                 TENSOR(Input1, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(i8symm_i8symm_i8symm_const1,
                 TENSOR(Input0, INT8, SYMM, CONSTANT),
                 TENSOR(Input1, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(u8asymm_u8asymm_u8asymm_const1,
                 TENSOR(Input0, UINT8, ASYMM, CONSTANT),
                 TENSOR(Input1, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(f32_f32_f32_const2,
                 TENSOR(Input0, FP32, NO_QUANT),
                 TENSOR(Input1, FP32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(f16_f16_f16_const2,
                 TENSOR(Input0, FP16, NO_QUANT),
                 TENSOR(Input1, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(i32_i32_i32_const2,
                 TENSOR(Input0, INT32, NO_QUANT),
                 TENSOR(Input1, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT32, NO_QUANT),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(i8asymm_i8asymm_i8asymm_const2,
                 TENSOR(Input0, INT8, ASYMM),
                 TENSOR(Input1, INT8, ASYMM, CONSTANT),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(i8symm_i8symm_i8symm_const2,
                 TENSOR(Input0, INT8, SYMM),
                 TENSOR(Input1, INT8, SYMM, CONSTANT),
                 TENSOR(Output, INT8, SYMM),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(u8asymm_u8asymm_u8asymm_const2,
                 TENSOR(Input0, UINT8, ASYMM),
                 TENSOR(Input1, UINT8, ASYMM, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Activation, INT32))

ADD_SIGNATURE(f32_f32_f32,
              f16_f16_f16,
              i32_i32_i32,
              i8asymm_i8asymm_i8asymm,
              i8symm_i8symm_i8symm,
              u8asymm_u8asymm_u8asymm,
              f32_f32_f32_const1,
              f16_f16_f16_const1,
              i32_i32_i32_const1,
              i8asymm_i8asymm_i8asymm_const1,
              i8symm_i8symm_i8symm_const1,
              u8asymm_u8asymm_u8asymm_const1,
              f32_f32_f32_const2,
              f16_f16_f16_const2,
              i32_i32_i32_const2,
              i8asymm_i8asymm_i8asymm_const2,
              i8symm_i8symm_i8symm_const2,
              u8asymm_u8asymm_u8asymm_const2)