START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Input2, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Adj_x, BOOL8),
                 SCALAR(Adj_y, BOOL8))

DEFINE_SIGNATURE(i32_i32,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Input2, INT32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT),
                 SCALAR(Adj_x, BOOL8),
                 SCALAR(Adj_y, BOOL8))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Input2, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Adj_x, BOOL8),
                 SCALAR(Adj_y, BOOL8))

DEFINE_SIGNATURE(i8asym_i8asym,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Input2, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(Adj_x, BOOL8),
                 SCALAR(Adj_y, BOOL8))

ADD_SIGNATURE(f32_f32,
              i32_i32,
              f16_f16,
              i8asym_i8asym)