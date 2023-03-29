START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32_f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Weight, FP32, NO_QUANT, CONSTANT),
                 TENSOR(Bias, FP32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_f16_f32_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Weight, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Bias, FP32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(f16_f16_f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Weight, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Bias, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(f16_f16_f32_bf16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Weight, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Bias, FP32, NO_QUANT, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(f16_f16_f32_f32,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Weight, FP16, NO_QUANT, CONSTANT),
                 TENSOR(Bias, FP32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(bf16_bf16_bf16_f16,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Weight, BF16, NO_QUANT, CONSTANT),
                 TENSOR(Bias, BF16, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(bf16_bf16_bf16_bf16,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Weight, BF16, NO_QUANT, CONSTANT),
                 TENSOR(Bias, BF16, NO_QUANT, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(bf16_bf16_bf16_f32,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Weight, BF16, NO_QUANT, CONSTANT),
                 TENSOR(Bias, BF16, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f32_bf16_f32_f16,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Weight, BF16, NO_QUANT, CONSTANT),
                 TENSOR(Bias, FP32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(f32_bf16_f32_bf16,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Weight, BF16, NO_QUANT, CONSTANT),
                 TENSOR(Bias, FP32, NO_QUANT, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(f32_bf16_f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Weight, BF16, NO_QUANT, CONSTANT),
                 TENSOR(Bias, FP32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(i8dfp_i8dfp_i8dfp_i8dfp,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Weight, INT8, DFP, CONSTANT),
                 TENSOR(Bias, INT8, DFP, CONSTANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(i16dfp_i16dfp_i16dfp_i16dfp,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Weight, INT16, DFP, CONSTANT),
                 TENSOR(Bias, INT16, DFP, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(i16dfp_i16dfp_i64dfp_f16,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Weight, INT16, DFP, CONSTANT),
                 TENSOR(Bias, INT64, DFP, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(u8asym_u8asym_i32asym_u8asym,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asym_u8asym_i32asym_i16dfp,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(u8asym_u8asym_i32asym_f16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(u8asym_u8pcq_i32pcq_u8asym,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, UINT8, SYMM_PCQ, CONSTANT),
                 TENSOR(Bias, INT32, SYMM_PCQ, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

/* HW 9.0.1 */
DEFINE_SIGNATURE(u8asym_u8asym_i32asym_i8dfp,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(u8asym_u8asym_i32asym_i16dfp,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(u8asym_u8asym_i32asym_bf16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(u8asym_u8asym_i32asym_f32,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(u8asym_u8pcq_i32pcq_i8dfp,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, UINT8, SYMM_PCQ, CONSTANT),
                 TENSOR(Bias, INT32, SYMM_PCQ, CONSTANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(u8asym_u8pcq_i32pcq_i16dfp,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, UINT8, SYMM_PCQ, CONSTANT),
                 TENSOR(Bias, INT32, SYMM_PCQ, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(u8asym_u8pcq_i32pcq_f16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, UINT8, SYMM_PCQ, CONSTANT),
                 TENSOR(Bias, INT32, SYMM_PCQ, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(u8asym_u8pcq_i32pcq_bf16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, UINT8, SYMM_PCQ, CONSTANT),
                 TENSOR(Bias, INT32, SYMM_PCQ, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(u8asym_u8pcq_i32pcq_f32,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, UINT8, SYMM_PCQ, CONSTANT),
                 TENSOR(Bias, INT32, SYMM_PCQ, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(u8asym_i8dfp_i32dfp_u8asym,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, INT8, DFP, CONSTANT),
                 TENSOR(Bias, INT32, DFP, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asym_i8dfp_i32dfp_i8dfp,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, INT8, DFP, CONSTANT),
                 TENSOR(Bias, INT32, DFP, CONSTANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(u8asym_i8dfp_i32dfp_i16dfp,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, INT8, DFP, CONSTANT),
                 TENSOR(Bias, INT32, DFP, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(u8asym_i8dfp_i32dfp_f16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, INT8, DFP, CONSTANT),
                 TENSOR(Bias, INT32, DFP, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(u8asym_i8dfp_i32dfp_bf16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, INT8, DFP, CONSTANT),
                 TENSOR(Bias, INT32, DFP, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(u8asym_i8dfp_i32dfp_f32,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, INT8, DFP, CONSTANT),
                 TENSOR(Bias, INT32, DFP, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(u8asym_i8pcq_i32pcq_u8asym,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, INT8, SYMM_PCQ, CONSTANT),
                 TENSOR(Bias, INT32, SYMM_PCQ, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asym_i8pcq_i32pcq_i8dfp,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, INT8, SYMM_PCQ, CONSTANT),
                 TENSOR(Bias, INT32, SYMM_PCQ, CONSTANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(u8asym_i8pcq_i32pcq_i16dfp,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, INT8, SYMM_PCQ, CONSTANT),
                 TENSOR(Bias, INT32, SYMM_PCQ, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(u8asym_i8pcq_i32pcq_f16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, INT8, SYMM_PCQ, CONSTANT),
                 TENSOR(Bias, INT32, SYMM_PCQ, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(u8asym_i8pcq_i32pcq_bf16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, INT8, SYMM_PCQ, CONSTANT),
                 TENSOR(Bias, INT32, SYMM_PCQ, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(u8asym_i8pcq_i32pcq_f32,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Weight, INT8, SYMM_PCQ, CONSTANT),
                 TENSOR(Bias, INT32, SYMM_PCQ, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(i8dfp_u8asym_i32asym_u8asym,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(i8dfp_u8asym_i32asym_i8dfp,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(i8dfp_u8asym_i32asym_i16dfp,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(i8dfp_u8asym_i32asym_f16,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i8dfp_u8asym_i32asym_bf16,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(i8dfp_u8asym_i32asym_f32,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(i8dfp_i8dfp_i32dfp_u8asym,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Weight, INT8, DFP, CONSTANT),
                 TENSOR(Bias, INT32, DFP, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(i8dfp_i8dfp_i32dfp_i16dfp,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Weight, INT8, DFP, CONSTANT),
                 TENSOR(Bias, INT32, DFP, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(i8dfp_i8dfp_i32dfp_bf16,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Weight, INT8, DFP, CONSTANT),
                 TENSOR(Bias, INT32, DFP, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(i8dfp_i8dfp_i32dfp_f32,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Weight, INT8, DFP, CONSTANT),
                 TENSOR(Bias, INT32, DFP, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(i8sym_u8asym_i32asym_u8asym,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(i8sym_u8asym_i32asym_i8dfp,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(i8sym_u8asym_i32asym_i16dfp,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(i8sym_u8asym_i32asymf16,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i8sym_u8asym_i32asym_bf16,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(i8sym_u8asym_i32asym_f32,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(i8sym_i8asym_i32asym_u8asym,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Weight, INT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(i8sym_i8asym_i32asym_i8dfp,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Weight, INT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(i8sym_i8asym_i32asym_i16dfp,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Weight, INT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(i8sym_i8asym_i32asym_bf16,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Weight, INT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(i8sym_i8asym_i32asym_f32,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Weight, INT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(i8asym_u8asym_i32asym_u8asym,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(i8asym_u8asym_i32asym_i8dfp,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(i8asym_u8asym_i32asym_i16dfp,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(i8asym_u8asym_i32asymf16,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i8asym_u8asym_i32asym_bf16,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(i8asym_u8asym_i32asym_f32,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Weight, UINT8, ASYMM, CONSTANT),
                 TENSOR(Bias, INT32, ASYMM, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))


ADD_SIGNATURE(f32_f32_f32_f32,
              f16_f16_f32_f16,
              f16_f16_f16_f16,
              f16_f16_f32_bf16,
              f16_f16_f32_f32,
              bf16_bf16_bf16_f16,
              bf16_bf16_bf16_bf16,
              bf16_bf16_bf16_f32,
              f32_bf16_f32_f16,
              f32_bf16_f32_bf16,
              f32_bf16_f32_f32,
              i8dfp_i8dfp_i8dfp_i8dfp,
              i16dfp_i16dfp_i16dfp_i16dfp,
              i16dfp_i16dfp_i64dfp_f16,
              u8asym_u8asym_i32asym_u8asym,
              u8asym_u8asym_i32asym_i16dfp,
              u8asym_u8asym_i32asym_f16,
              u8asym_u8pcq_i32pcq_u8asym,
              u8asym_u8asym_i32asym_i8dfp,
              u8asym_u8asym_i32asym_i16dfp,
              u8asym_u8asym_i32asym_bf16,
              u8asym_u8asym_i32asym_f32,
              u8asym_u8pcq_i32pcq_i8dfp,
              u8asym_u8pcq_i32pcq_i16dfp,
              u8asym_u8pcq_i32pcq_f16,
              u8asym_u8pcq_i32pcq_bf16,
              u8asym_u8pcq_i32pcq_f32,
              u8asym_i8dfp_i32dfp_u8asym,
              u8asym_i8dfp_i32dfp_i8dfp,
              u8asym_i8dfp_i32dfp_i16dfp,
              u8asym_i8dfp_i32dfp_f16,
              u8asym_i8dfp_i32dfp_bf16,
              u8asym_i8dfp_i32dfp_f32,
              u8asym_i8pcq_i32pcq_u8asym,
              u8asym_i8pcq_i32pcq_i8dfp,
              u8asym_i8pcq_i32pcq_i16dfp,
              u8asym_i8pcq_i32pcq_f16,
              u8asym_i8pcq_i32pcq_bf16,
              u8asym_i8pcq_i32pcq_f32,
              i8dfp_u8asym_i32asym_u8asym,
              i8dfp_u8asym_i32asym_i8dfp,
              i8dfp_u8asym_i32asym_i16dfp,
              i8dfp_u8asym_i32asym_f16,
              i8dfp_u8asym_i32asym_bf16,
              i8dfp_u8asym_i32asym_f32,
              i8dfp_i8dfp_i32dfp_u8asym,
              i8dfp_i8dfp_i32dfp_i16dfp,
              i8dfp_i8dfp_i32dfp_bf16,
              i8dfp_i8dfp_i32dfp_f32,
              i8sym_u8asym_i32asym_u8asym,
              i8sym_u8asym_i32asym_i8dfp,
              i8sym_u8asym_i32asym_i16dfp,
              i8sym_u8asym_i32asymf16,
              i8sym_u8asym_i32asym_bf16,
              i8sym_u8asym_i32asym_f32,
              i8sym_i8asym_i32asym_u8asym,
              i8sym_i8asym_i32asym_i8dfp,
              i8sym_i8asym_i32asym_i16dfp,
              i8sym_i8asym_i32asym_bf16,
              i8sym_i8asym_i32asym_f32,
              i8asym_u8asym_i32asym_u8asym,
              i8asym_u8asym_i32asym_i8dfp,
              i8asym_u8asym_i32asym_i16dfp,
              i8asym_u8asym_i32asymf16,
              i8asym_u8asym_i32asym_bf16,
              i8asym_u8asym_i32asym_f32)