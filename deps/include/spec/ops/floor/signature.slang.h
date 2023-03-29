START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f32_f16,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(f16_u8asym,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(f16_i16dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(f16_i16asym,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT16, ASYMM))

DEFINE_SIGNATURE(f16_i16sym,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT16, SYMM))

DEFINE_SIGNATURE(f16_i8dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(f16_i8asym,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(f16_i8sym,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Output, INT8, SYMM))

DEFINE_SIGNATURE(u8asymm_f16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(u8asymm_u8asym,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(i16dfp_f16,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i16asym_f16,
                 TENSOR(Input, INT16, ASYMM),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i16sym_f16,
                 TENSOR(Input, INT16, SYMM),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i16dfp_i16dfp,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(i16asym_i16asym,
                 TENSOR(Input, INT16, ASYMM),
                 TENSOR(Output, INT16, ASYMM))

DEFINE_SIGNATURE(i16sym_i16sym,
                 TENSOR(Input, INT16, SYMM),
                 TENSOR(Output, INT16, SYMM))

DEFINE_SIGNATURE(i8dfp_f16,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i8asym_f16,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i8sym_f16,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i8dfp_i8dfp,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(i8asym_i8asym,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(i8sym_i8sym,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Output, INT8, SYMM))

ADD_SIGNATURE(f32_f32,
              f32_f16,
              f16_f32,
              f16_f16,
              f16_u8asym,
              f16_i16dfp,
              f16_i16asym,
              f16_i16sym,
              f16_i8dfp,
              f16_i8asym,
              f16_i8sym,
              u8asymm_f16,
              u8asymm_u8asym,
              i16dfp_f16,
              i16asym_f16,
              i16sym_f16,
              i16dfp_i16dfp,
              i16asym_i16asym,
              i16sym_i16sym,
              i8dfp_f16,
              i8asym_f16,
              i8sym_f16,
              i8dfp_i8dfp,
              i8asym_i8asym,
              i8sym_i8sym)