START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Regions, FP32, NO_QUANT),
                 TENSOR(BatchIndex, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(OutputHeight, INT32),
                 SCALAR(OutputWidth, INT32),
                 SCALAR(HeightRatio, FP32),
                 SCALAR(WidthRatio, FP32),
                 SCALAR(HSampleNum, INT32),
                 SCALAR(WSampleNum, INT32),
                 SCALAR(Layout, BOOL8))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Regions, FP16, NO_QUANT),
                 TENSOR(BatchIndex, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(OutputHeight, INT32),
                 SCALAR(OutputWidth, INT32),
                 SCALAR(HeightRatio, FP16),
                 SCALAR(WidthRatio, FP16),
                 SCALAR(HSampleNum, INT32),
                 SCALAR(WSampleNum, INT32),
                 SCALAR(Layout, BOOL8))

DEFINE_SIGNATURE(u8asymm_u8asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Regions, UINT16, ASYMM),
                 TENSOR(BatchIndex, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(OutputHeight, INT32),
                 SCALAR(OutputWidth, INT32),
                 SCALAR(HeightRatio, FP32),
                 SCALAR(WidthRatio, FP32),
                 SCALAR(HSampleNum, INT32),
                 SCALAR(WSampleNum, INT32),
                 SCALAR(Layout, BOOL8))

DEFINE_SIGNATURE(i8asymm_i8asymm,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Regions, UINT16, ASYMM),
                 TENSOR(BatchIndex, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(OutputHeight, INT32),
                 SCALAR(OutputWidth, INT32),
                 SCALAR(HeightRatio, FP32),
                 SCALAR(WidthRatio, FP32),
                 SCALAR(HSampleNum, INT32),
                 SCALAR(WSampleNum, INT32),
                 SCALAR(Layout, BOOL8))

ADD_SIGNATURE(f32_f32,
              f16_f16,
              u8asymm_u8asymm,
              i8asymm_i8asymm)