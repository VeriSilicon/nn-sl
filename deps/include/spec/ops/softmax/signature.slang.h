START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(f32_f16,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Beta, FP16),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(f16_f32,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Beta, FP16),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(f16_i16dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT16, DFP),
                 SCALAR(Beta, FP16),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(f16_i16asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT16, ASYMM),
                 SCALAR(Beta, FP16),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(f16_i16symm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT16, SYMM),
                 SCALAR(Beta, FP16),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(f16_i8dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT8, DFP),
                 SCALAR(Beta, FP16),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(f16_i8symm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT8, SYMM),
                 SCALAR(Beta, FP16),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(f16_i8asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(Beta, FP16),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(f16_u8asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Beta, FP16),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(bf16_bf16,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Output, BF16, NO_QUANT),
                 SCALAR(Beta, FP16),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(bf16_f32,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Beta, FP16),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(bf16_fp16,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Beta, FP16),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(u8asymm_u8asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(u8asymm_f16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(u8asymm_f32,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i8asymm_i8asymm,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i8symm_i8symm,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i8asymm_f16,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i8symm_f16,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i8asymm_f32,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i8symm_f32,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i8dfp_i8dfp,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Output, INT8, DFP),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i8dfp_f16,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i8dfp_f32,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i16asymm_i16asymm,
                 TENSOR(Input, INT16, ASYMM),
                 TENSOR(Output, INT16, ASYMM),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i16symm_i16symm,
                 TENSOR(Input, INT16, SYMM),
                 TENSOR(Output, INT16, SYMM),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i16asymm_f16,
                 TENSOR(Input, INT16, ASYMM),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i16symm_f16,
                 TENSOR(Input, INT16, SYMM),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i16asymm_f32,
                 TENSOR(Input, INT16, ASYMM),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i16symm_f32,
                 TENSOR(Input, INT16, SYMM),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i16dfp_f32,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i16dfp_f16,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

DEFINE_SIGNATURE(i16dfp_i16dfp,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Output, INT16, DFP),
                 SCALAR(Beta, FP32),
                 SCALAR(Axis, INT32))

ADD_SIGNATURE(f32_f32,
              f32_f16,
              f16_f16,
              f16_f32,
              f16_i16dfp,
              f16_i16asymm,
              f16_i16symm,
              f16_i8dfp,
              f16_i8symm,
              f16_i8asymm,
              f16_u8asymm,
              bf16_bf16,
              bf16_f32,
              bf16_fp16,

              u8asymm_u8asymm,
              u8asymm_f16,
              u8asymm_f32,
              i8asymm_i8asymm,
              i8symm_i8symm,
              i8asymm_f16,
              i8symm_f16,
              i8asymm_f32,
              i8symm_f32,
              i8dfp_i8dfp,
              i8dfp_f16,
              i8dfp_f32,
              i16asymm_i16asymm,
              i16symm_i16symm,
              i16asymm_f16,
              i16symm_f16,
              i16asymm_f32,
              i16symm_f32,
              i16dfp_f32,
              i16dfp_f16,
              i16dfp_i16dfp)