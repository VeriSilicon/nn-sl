START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Axis, INT32),
                 SCALAR(Input_cnt, INT32))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Axis, INT32),
                 SCALAR(Input_cnt, INT32))

DEFINE_SIGNATURE(i32_i32,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT),
                 SCALAR(Axis, INT32),
                 SCALAR(Input_cnt, INT32))

DEFINE_SIGNATURE(u8asym_u8asym,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Axis, INT32),
                 SCALAR(Input_cnt, INT32))

DEFINE_SIGNATURE(i8asym_i8asym,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(Axis, INT32),
                 SCALAR(Input_cnt, INT32))

DEFINE_SIGNATURE(i8sym_i8sym,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM),
                 SCALAR(Axis, INT32),
                 SCALAR(Input_cnt, INT32))

ADD_SIGNATURE(f32_f32,
              f16_f16,
              i32_i32,
              u8asym_u8asym,
              i8asym_i8asym,
              i8sym_i8sym)