START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Alpha, FP16))

DEFINE_SIGNATURE(f16_i16dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT16, DFP),
                 SCALAR(Alpha, FP16))

DEFINE_SIGNATURE(f16_i16asym,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT16, ASYMM),
                 SCALAR(Alpha, FP16))

DEFINE_SIGNATURE(f16_i16sym,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT16, SYMM),
                 SCALAR(Alpha, FP16))

DEFINE_SIGNATURE(f16_i8dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT8, DFP),
                 SCALAR(Alpha, FP16))

DEFINE_SIGNATURE(f16_i8asym,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(Alpha, FP16))

DEFINE_SIGNATURE(f16_i8sym,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT8, SYMM),
                 SCALAR(Alpha, FP16))

DEFINE_SIGNATURE(f16_u8asym,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Alpha, FP16))

DEFINE_SIGNATURE(i8dfp_f16,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i8asym_f16,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i8sym_f16,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i8dfp_i8dfp,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Output, INT8, DFP),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i8asym_i8asym,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i8sym_i8sym,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(u8asym_u8asym,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i16dfp_f16,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i16asym_f16,
                 TENSOR(Input, INT16, ASYMM),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i16sym_f16,
                 TENSOR(Input, INT16, SYMM),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i16dfp_i16dfp,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Output, INT16, DFP),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i16asym_i16asym,
                 TENSOR(Input, INT16, ASYMM),
                 TENSOR(Output, INT16, ASYMM),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i16sym_i16sym,
                 TENSOR(Input, INT16, SYMM),
                 TENSOR(Output, INT16, SYMM),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(bf16_bf16,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Output, BF16, NO_QUANT),
                 SCALAR(Alpha, BF16))

DEFINE_SIGNATURE(bf16_f32,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Alpha, BF16))

DEFINE_SIGNATURE(f32_bf16,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, BF16, NO_QUANT),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(f32_u8asymm,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(u8asymm_f32,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Alpha, FP32))

/* HW 9.0.1 */
DEFINE_SIGNATURE(u8asymm_i8dfp,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, INT8, DFP),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(u8asymm_i16dfp,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, INT16, DFP),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(u8asymm_i8dfp,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, INT8, DFP),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(u8asymm_f16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, FP16, NO_QUANT),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i8dfp_u8asym,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i8dfp_i16dfp,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Output, INT16, DFP),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i8dfp_bf16,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Output, BF16, NO_QUANT),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i8dfp_f32,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i16dfp_u8asym,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Output, UINT8, ASYMM),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i16dfp_i8dfp,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Output, INT8, DFP),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i16dfp_bf16,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Output, BF16, NO_QUANT),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i16dfp_f32,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(f16_bf16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, BF16, NO_QUANT),
                 SCALAR(Alpha, FP16))

DEFINE_SIGNATURE(f16_f32,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT),
                 SCALAR(Alpha, FP16))

/* HW 9.1.1 */
DEFINE_SIGNATURE(u4asym_u4asym,
                 TENSOR(Input, UINT4, ASYMM),
                 TENSOR(Output, UINT4, ASYMM),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(u4sym_u4sym,
                 TENSOR(Input, UINT4, SYMM),
                 TENSOR(Output, UINT4, SYMM),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i4asym_i4asym,
                 TENSOR(Input, INT4, ASYMM),
                 TENSOR(Output, INT4, ASYMM),
                 SCALAR(Alpha, FP32))

DEFINE_SIGNATURE(i4sym_i4sym,
                 TENSOR(Input, INT4, SYMM),
                 TENSOR(Output, INT4, SYMM),
                 SCALAR(Alpha, FP32))

ADD_SIGNATURE(f32_f32,
              f16_f16,
              f16_i16dfp,
              f16_i16asym,
              f16_i16sym,
              f16_i8dfp,
              f16_i8asym,
              f16_i8sym,
              f16_u8asym,
              i8dfp_f16,
              i8asym_f16,
              i8sym_f16,
              i8dfp_i8dfp,
              i8asym_i8asym,
              i8sym_i8sym,
              u8asym_u8asym,
              i16dfp_f16,
              i16asym_f16,
              i16sym_f16,
              i16dfp_i16dfp,
              i16asym_i16asym,
              i16sym_i16sym,
              bf16_bf16,
              bf16_f32,
              f32_bf16,
              f32_u8asymm,
              u8asymm_f32,
              u8asymm_i8dfp,
              u8asymm_i16dfp,
              u8asymm_i8dfp,
              u8asymm_f16,
              i8dfp_u8asym,
              i8dfp_i16dfp,
              i8dfp_bf16,
              i8dfp_f32,
              i16dfp_u8asym,
              i16dfp_i8dfp,
              i16dfp_bf16,
              i16dfp_f32,
              f16_bf16,
              f16_f32,
              u4asym_u4asym,
              u4sym_u4sym,
              i4asym_i4asym,
              i4sym_i4sym)