START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(f32_f32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_f16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_f32,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(f16_i32,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(f16_u32,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT32, NO_QUANT))

DEFINE_SIGNATURE(f16_bf16,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(f16_i16dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(f16_i16asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT16, ASYMM))

DEFINE_SIGNATURE(f16_i16symm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT16, SYMM))

DEFINE_SIGNATURE(f16_i8dfp,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(f16_i8asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(f16_i8symm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, SYMM))

DEFINE_SIGNATURE(f16_u8asymm,
                 TENSOR(Input, FP16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(f32_i32dfp,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT32, DFP))

DEFINE_SIGNATURE(f32_i32asymm,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT32, ASYMM))

DEFINE_SIGNATURE(f32_u32,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT32, NO_QUANT))

DEFINE_SIGNATURE(f32_f16,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(f32_bf16,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(f32_i16dfp,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(f32_i8dfp,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(f32_u8asymm,
                 TENSOR(Input, FP32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(i16dfp_f32,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(i16dfp_i32,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(i16dfp_u32,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT32, NO_QUANT))

DEFINE_SIGNATURE(i16dfp_i16dfp,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(i16asymm_i16asymm,
                 TENSOR(Input, INT16, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT16, ASYMM))

DEFINE_SIGNATURE(i16symm_i16symm,
                 TENSOR(Input, INT16, SYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT16, SYMM))

DEFINE_SIGNATURE(i16dfp_i8dfp,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(i16dfp_u8asymm,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(i16dfp_f16,
                 TENSOR(Input, INT16, DFP),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i16asymm_f16,
                 TENSOR(Input, INT16, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i16symm_f16,
                 TENSOR(Input, INT16, SYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i8dfp_f32,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(i8symm_f16,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i8asymm_f16,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i8dfp_f16,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(i8dfp_i32dfp,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT32, DFP))

DEFINE_SIGNATURE(i8dfp_u32,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT32, NO_QUANT))

DEFINE_SIGNATURE(i8dfp_i8dfp,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(i8dfp_i8asymm,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(i8dfp_i16dfp,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(i8dfp_u8asymm,
                 TENSOR(Input, INT8, DFP),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(i8asymm_i8asymm,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(i8asymm_i8dfp,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(i8symm_i8symm,
                 TENSOR(Input, INT8, SYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, SYMM))

DEFINE_SIGNATURE(i8asymm_u8asymm,
                 TENSOR(Input, INT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asymm_u8asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asymm_i8asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, ASYMM))

DEFINE_SIGNATURE(u8asymm_f16,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(u8asymm_i8dfp,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(u8asymm_i16dfp,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(u8asymm_i32asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT32, ASYMM))

DEFINE_SIGNATURE(u8asymm_u32,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT32, NO_QUANT))

DEFINE_SIGNATURE(u8asymm_f32,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(bf16_bf16,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(bf16_f16,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP16, NO_QUANT))

DEFINE_SIGNATURE(bf16_f32,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, FP32, NO_QUANT))

DEFINE_SIGNATURE(i32_i32,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(i32_i16dfp,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(i32_i8dfp,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(i32_u32,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT32, NO_QUANT))

DEFINE_SIGNATURE(i32_u16,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT16, NO_QUANT))

DEFINE_SIGNATURE(i32_u8asymm,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u32_u32,
                 TENSOR(Input, UINT32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT32, NO_QUANT))

DEFINE_SIGNATURE(u32_i16dfp,
                 TENSOR(Input, UINT32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT16, DFP))

DEFINE_SIGNATURE(u32_i8dfp,
                 TENSOR(Input, UINT32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT8, DFP))

DEFINE_SIGNATURE(u32_i32,
                 TENSOR(Input, UINT32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(u32_u8asymm,
                 TENSOR(Input, UINT32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u32_u8,
                 TENSOR(Input, UINT32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, NO_QUANT))

DEFINE_SIGNATURE(bf16_i32,
                 TENSOR(Input, BF16, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT32, NO_QUANT))

DEFINE_SIGNATURE(i32_bf16,
                 TENSOR(Input, INT32, NO_QUANT),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, BF16, NO_QUANT))

DEFINE_SIGNATURE(u4asymm_u8asymm,
                 TENSOR(Input, UINT4, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u4symm_u8asymm,
                 TENSOR(Input, UINT4, SYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asymm_u4asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT4, ASYMM))

DEFINE_SIGNATURE(u8asymm_u4symm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT4, SYMM))

DEFINE_SIGNATURE(i4asymm_u8asymm,
                 TENSOR(Input, INT4, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(i4symm_u8asymm,
                 TENSOR(Input, INT4, SYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, UINT8, ASYMM))

DEFINE_SIGNATURE(u8asymm_i4asymm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT4, ASYMM))

DEFINE_SIGNATURE(u8asymm_i4symm,
                 TENSOR(Input, UINT8, ASYMM),
                 TENSOR(Shape, INT32, NO_QUANT, CONSTANT),
                 TENSOR(Output, INT4, SYMM))

ADD_SIGNATURE(f32_f32,
              f16_f16,
              f16_f32,
              f16_i32,
              f16_u32,
              f16_bf16,
              f16_i16dfp,
              f16_i16asymm,
              f16_i16asymm,
              f16_i16symm,
              f16_i8dfp,
              f16_i8asymm,
              f16_i8symm,
              f16_u8asymm,
              f32_i32dfp,
              f32_i32asymm,
              f32_u32,
              f32_f16,
              f32_bf16,
              f32_i16dfp,
              f32_i8dfp,
              f32_u8asymm,
              i16dfp_f32,
              i16dfp_i32,
              i16dfp_u32,
              i16dfp_i16dfp,
              i16asymm_i16asymm,
              i16symm_i16symm,
              i16dfp_i8dfp,
              i16dfp_u8asymm,
              i16dfp_f16,
              i16asymm_f16,
              i16symm_f16,
              i8dfp_f32,
              i8symm_f16,
              i8asymm_f16,
              i8dfp_f16,
              i8dfp_i32dfp,
              i8dfp_u32,
              i8dfp_i8dfp,
              i8dfp_i8asymm,
              i8dfp_i16dfp,
              i8dfp_u8asymm,
              i8asymm_i8asymm,
              i8asymm_i8dfp,
              i8symm_i8symm,
              i8asymm_u8asymm,
              u8asymm_u8asymm,
              u8asymm_i8asymm,
              u8asymm_f16,
              u8asymm_i8dfp,
              bf16_bf16,
              u8asymm_i32asymm,
              u8asymm_u32,
              u8asymm_f32,
              u8asymm_i16dfp,
              bf16_f16,
              bf16_f32,
              i32_i32,
              i32_i16dfp,
              i32_i8dfp,
              i32_u32,
              i32_u16,
              i32_u8asymm,
              u32_u32,
              u32_i16dfp,
              u32_i8dfp,
              u32_i32,
              u32_u8asymm,
              u32_u8,
              bf16_i32,
              i32_bf16,
              u4asymm_u8asymm,
              u4symm_u8asymm,
              u8asymm_u4asymm,
              u8asymm_u4symm,
              i4asymm_u8asymm,
              i4symm_u8asymm,
              u8asymm_i4asymm,
              u8asymm_i4symm)