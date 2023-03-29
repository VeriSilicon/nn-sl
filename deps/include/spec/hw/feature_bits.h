#pragma once
#include <cstdint>

#define DEF_FEATURE(ftype, fname) \
  static constexpr ftype fname = PID::fname;

#define INIT_FEATURE(ftype, fname, value) \
  static constexpr ftype fname = (value);

namespace hw {
namespace spec {
template <typename PID>
struct feature_bits { // naming aligned with gc_feature_database.h

  // IP information
  DEF_FEATURE(uint32_t, chipID)
  DEF_FEATURE(uint32_t, chipVersion)
  DEF_FEATURE(uint32_t, productID)
  DEF_FEATURE(uint32_t, ecoID)
  DEF_FEATURE(uint32_t, customerID)
  DEF_FEATURE(uint32_t, patchVersion)

  // neural network engine feature
  DEF_FEATURE(uint32_t, NNCoreCount)
  DEF_FEATURE(uint32_t, NNCoreCount_INT16)
  DEF_FEATURE(uint32_t, NNCoreCount_INT8)
  DEF_FEATURE(uint32_t, NNCoreCount_FLOAT16)
  DEF_FEATURE(uint32_t, NNCoreCount_BFLOAT)
  DEF_FEATURE(uint32_t, NN_ASYMMETRIC_INT8)
  DEF_FEATURE(bool, NN_PER_CHANNEL_QUANT)
  DEF_FEATURE(bool, NN_PER_CHANNEL_QUANT_ASYM)
  DEF_FEATURE(bool, NN_XYDP0)
  DEF_FEATURE(bool, NN_XYDP9)
  DEF_FEATURE(bool, NN_ZDP6)
  DEF_FEATURE(bool, NN_DEPTHWISE_SUPPORT)
  DEF_FEATURE(bool, NN_4BIT_PHASE1)
  DEF_FEATURE(bool, NN_FLOAT32_IO)
  DEF_FEATURE(bool, VIP_V7)
  DEF_FEATURE(bool, TF_QUANTIZATION)
  DEF_FEATURE(bool, NN_POST_OUT_SUPPORT_FP32)
  DEF_FEATURE(bool, NN_POST_OUT_SUPPORT_FP16)
  DEF_FEATURE(bool, NN_POST_OUT_SUPPORT_BF16)
  DEF_FEATURE(bool, OUTPUT_CONVERT_UINT8_INT8_TO_UINT16_INT16_FIX)
  DEF_FEATURE(bool, NN_SUPPORT_16_8_QUANTIZATION)
  DEF_FEATURE(bool, NN_POST_MULT_SUPPORT_FP_CONV)


  // tensor processor engine features
  DEF_FEATURE(uint32_t, TPEngine_CoreCount)
  DEF_FEATURE(bool, TP_FLOAT32_IO)
  DEF_FEATURE(uint32_t, TPLite_CoreCount)

  // ppu features
  DEF_FEATURE(uint32_t, NumShaderCores)
  DEF_FEATURE(uint32_t, ThreadCount)
  DEF_FEATURE(bool, EVIS_VX2)
};
}  // namespace spec
}  // namespace hw
#undef DEF_FEATURE
