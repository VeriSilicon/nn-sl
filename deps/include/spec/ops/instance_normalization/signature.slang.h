START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Epsilon, FP32),
                 SCALAR(Layout, BOOL8))

DEFINE_SIGNATURE(f16_f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Epsilon, FP16),
                 SCALAR(Layout, BOOL8))

ADD_SIGNATURE(f32_f32_f32,
              f16_f16_f16)