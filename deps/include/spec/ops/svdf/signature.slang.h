START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(WeightsFeature, FP32, NO_QUANT),
                 TENSOR(WeightsTime, FP32, NO_QUANT),
                 TENSOR(Bias, FP32, NO_QUANT),
                 TENSOR(StateIn, FP32, NO_QUANT),
                 TENSOR(StateOut, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Rank, INT32),
                 SCALAR(NumUnits, INT32),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(WeightsFeature, FP16, NO_QUANT),
                 TENSOR(WeightsTime, FP16, NO_QUANT),
                 TENSOR(Bias, FP16, NO_QUANT),
                 TENSOR(StateIn, FP16, NO_QUANT),
                 TENSOR(StateOut, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Rank, INT32),
                 SCALAR(NumUnits, INT32),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(u8asymm_u8asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(WeightsFeature, UINT8, ASYMM),
                 TENSOR(WeightsTime, UINT8, ASYMM),
                 TENSOR(Bias, UINT8, ASYMM),
                 TENSOR(StateIn, UINT8, ASYMM),
                 TENSOR(StateOut, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Rank, INT32),
                 SCALAR(NumUnits, INT32),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(i8asymm_i8asymm,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(WeightsFeature, INT8, ASYMM),
                 TENSOR(WeightsTime, INT8, ASYMM),
                 TENSOR(Bias, INT8, ASYMM),
                 TENSOR(StateIn, INT8, ASYMM),
                 TENSOR(StateOut, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(Rank, INT32),
                 SCALAR(NumUnits, INT32),
                 SCALAR(Activation, INT32))

DEFINE_SIGNATURE(i8symm_i8symm,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(WeightsFeature, INT8, SYMM),
                 TENSOR(WeightsTime, INT8, SYMM),
                 TENSOR(Bias, INT8, SYMM),
                 TENSOR(StateIn, INT8, SYMM),
                 TENSOR(StateOut, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM),
                 SCALAR(Rank, INT32),
                 SCALAR(NumUnits, INT32),
                 SCALAR(Activation, INT32))

ADD_SIGNATURE(f32_f32,
              f16_f16,
              u8asymm_u8asymm,
              i8asymm_i8asymm,
              i8symm_i8symm)