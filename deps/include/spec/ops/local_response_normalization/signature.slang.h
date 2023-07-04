START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Radius, INT32),
                 SCALAR(Bias, FP32),
                 SCALAR(Alpha, FP32),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Radius, INT32),
                 SCALAR(Bias, FP16),
                 SCALAR(Alpha, FP16),
                 SCALAR(Beta, FP16),
                 SCALAR(Axis, INT32))

ADD_SIGNATURE(f32_f32,
              f16_f16)