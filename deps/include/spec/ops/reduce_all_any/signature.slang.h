START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(bool8_bool8,
                 TENSOR(Input, BOOL8, NO_QUANT),
                 TENSOR(Output, BOOL8, NO_QUANT),
                 TENSOR(Axis, INT32, NO_QUANT),
                 SCALAR(KeepDims, BOOL8))

DEFINE_SIGNATURE(i8_i8,
                 TENSOR(Input, INT8, NO_QUANT),
                 TENSOR(Output, INT8, NO_QUANT),
                 TENSOR(Axis, INT32, NO_QUANT),
                 SCALAR(KeepDims, BOOL8))

ADD_SIGNATURE(bool8_bool8,
              i8_i8)