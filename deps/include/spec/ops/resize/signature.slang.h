START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Output_width, INT32),
                 SCALAR(Output_height, INT32),
                 SCALAR(Factor, FP32),
                 SCALAR(Layout, BOOL8),
                 SCALAR(Align_corners, BOOL8),
                 SCALAR(Half_pixel_centers, BOOL8))

DEFINE_SIGNATURE(f16_f16_factor_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Output_width, INT32),
                 SCALAR(Output_height, INT32),
                 SCALAR(Factor, FP16),
                 SCALAR(Layout, BOOL8),
                 SCALAR(Align_corners, BOOL8),
                 SCALAR(Half_pixel_centers, BOOL8))

DEFINE_SIGNATURE(f16_f16_factor_f32,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Output_width, INT32),
                 SCALAR(Output_height, INT32),
                 SCALAR(Factor, FP32),
                 SCALAR(Layout, BOOL8),
                 SCALAR(Align_corners, BOOL8),
                 SCALAR(Half_pixel_centers, BOOL8))

DEFINE_SIGNATURE(i8asymm_i8asymm,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(Output_width, INT32),
                 SCALAR(Output_height, INT32),
                 SCALAR(Factor, FP32),
                 SCALAR(Layout, BOOL8),
                 SCALAR(Align_corners, BOOL8),
                 SCALAR(Half_pixel_centers, BOOL8))

DEFINE_SIGNATURE(i8symm_i8symm,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM),
                 SCALAR(Output_width, INT32),
                 SCALAR(Output_height, INT32),
                 SCALAR(Factor, FP32),
                 SCALAR(Layout, BOOL8),
                 SCALAR(Align_corners, BOOL8),
                 SCALAR(Half_pixel_centers, BOOL8))

DEFINE_SIGNATURE(u8asymm_u8asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Output_width, INT32),
                 SCALAR(Output_height, INT32),
                 SCALAR(Factor, FP32),
                 SCALAR(Layout, BOOL8),
                 SCALAR(Align_corners, BOOL8),
                 SCALAR(Half_pixel_centers, BOOL8))

ADD_SIGNATURE(f32_f32,
              f16_f16_factor_f16,
              f16_f16_factor_f32,
              i8asymm_i8asymm,
              i8symm_i8symm,
              u8asymm_u8asymm)