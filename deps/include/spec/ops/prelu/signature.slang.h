START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Alpha, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Alpha, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(f16_f16_u8asym,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Alpha, FP16, NO_QUANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(f16_f16_i16dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Alpha, FP16, NO_QUANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(f16_f16_i8dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Alpha, FP16, NO_QUANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(u8asym_f16_u8asym,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Alpha, FP16, NO_QUANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asym_f16_f16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Alpha, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i8dfp_f16_i8dfp,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Alpha, FP16, NO_QUANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(i8dfp_f16_f16,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Alpha, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i16dfp_f16_i16dfp,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Alpha, FP16, NO_QUANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(i16dfp_f16_f16,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Alpha, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(bf16_f16_bf16,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Alpha, FP16, NO_QUANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(bf16_bf16_bf16,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Alpha, BF16, NO_QUANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(i32_i32_i32,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Alpha, INT32, NO_QUANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(i8asymm_i8asymm_i8asymm,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Alpha, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(i8symm_i8symm_i8symm,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Alpha, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM))

DEFINE_SIGNATURE(u8asymm_u8asymm_u8asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Alpha, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asymm_u8asymm_f16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Alpha, UINT8, ASYMM),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(f32_f32_f32_const,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Alpha, FP32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_f16_f16_const,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Alpha, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(f16_f16_u8asym_const,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Alpha, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(f16_f16_i16dfp_const,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Alpha, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(f16_f16_i8dfp_const,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Alpha, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(u8asym_f16_u8asym_const,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Alpha, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asym_f16_f16_const,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Alpha, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i8dfp_f16_i8dfp_const,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Alpha, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(i8dfp_f16_f16_const,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Alpha, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i16dfp_f16_i16dfp_const,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Alpha, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(i16dfp_f16_f16_const,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Alpha, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(bf16_f16_bf16_const,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Alpha, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(bf16_bf16_bf16_const,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Alpha, BF16, NO_QUANT, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(i32_i32_i32_const,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Alpha, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(i8asymm_i8asymm_i8asymm_const,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Alpha, INT8, ASYMM, CONSTANT),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(i8symm_i8symm_i8symm_const,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Alpha, INT8, SYMM, CONSTANT),
                 TENSOR(Output, INT8, SYMM))

DEFINE_SIGNATURE(u8asymm_u8asymm_u8asymm_const,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Alpha, UINT8, ASYMM, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asymm_u8asymm_f16_const,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Alpha, UINT8, ASYMM, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

ADD_SIGNATURE(f32_f32_f32,
              f16_f16_f16,
              f16_f16_u8asym,
              f16_f16_i16dfp,
              f16_f16_i8dfp,
              u8asym_f16_u8asym,
              u8asym_f16_f16,
              i8dfp_f16_i8dfp,
              i8dfp_f16_f16,
              i16dfp_f16_i16dfp,
              i16dfp_f16_f16,
              bf16_f16_bf16,
              bf16_bf16_bf16,
              i32_i32_i32,
              i8asymm_i8asymm_i8asymm,
              i8symm_i8symm_i8symm,
              u8asymm_u8asymm_u8asymm,
              u8asymm_u8asymm_f16,
              f32_f32_f32_const,
              f16_f16_f16_const,
              f16_f16_u8asym_const,
              f16_f16_i16dfp_const,
              f16_f16_i8dfp_const,
              u8asym_f16_u8asym_const,
              u8asym_f16_f16_const,
              i8dfp_f16_i8dfp_const,
              i8dfp_f16_f16_const,
              i16dfp_f16_i16dfp_const,
              i16dfp_f16_f16_const,
              bf16_f16_bf16_const,
              bf16_bf16_bf16_const,
              i32_i32_i32_const,
              i8asymm_i8asymm_i8asymm_const,
              i8symm_i8symm_i8symm_const,
              u8asymm_u8asymm_u8asymm_const,
              u8asymm_u8asymm_f16_const)