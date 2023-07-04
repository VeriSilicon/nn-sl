START_DEFINE_SIGNATURES()

DEFINE_SIGNATURE(bool8_bool8,
                 TENSOR(Input, BOOL8, NO_QUANT),
                 TENSOR(Input1, BOOL8, NO_QUANT),
                 TENSOR(Output, BOOL8, NO_QUANT))

ADD_SIGNATURE(bool8_bool8)