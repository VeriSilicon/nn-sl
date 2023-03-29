#pragma once
#include "spec/hw/feature_bits.h"

struct VSI_VIP_HARDWARE {
  // IP information
  INIT_FEATURE(uint32_t, chipID, 0x0001)
  INIT_FEATURE(uint32_t, chipVersion, 0x0001)
  INIT_FEATURE(uint32_t, productID, 0x0000001)
  INIT_FEATURE(uint32_t, ecoID, 0x0000001)
  INIT_FEATURE(uint32_t, customerID, 0x00000001)
  INIT_FEATURE(uint32_t, patchVersion, 0x0)

  // neural network engine feature
  INIT_FEATURE(uint32_t, NNCoreCount, 0x10)
  INIT_FEATURE(uint32_t, NNCoreCount_INT16, 0x10)
  INIT_FEATURE(uint32_t, NNCoreCount_INT8, 0x10)
  INIT_FEATURE(uint32_t, NNCoreCount_FLOAT16, 0x10)
  INIT_FEATURE(uint32_t, NNCoreCount_BFLOAT, 0x10)
  INIT_FEATURE(uint32_t, NN_ASYMMETRIC_INT8, 0x10)
  INIT_FEATURE(bool, NN_PER_CHANNEL_QUANT, 0x10)
  INIT_FEATURE(bool, NN_PER_CHANNEL_QUANT_ASYM, 0x10)
  INIT_FEATURE(bool, NN_XYDP0, 0x10)
  INIT_FEATURE(bool, NN_XYDP9, 0x10)
  INIT_FEATURE(bool, NN_ZDP6, 0x10)
  INIT_FEATURE(bool, NN_DEPTHWISE_SUPPORT, 0x10)
  INIT_FEATURE(bool, NN_4BIT_PHASE1, 0x10)
  INIT_FEATURE(bool, NN_FLOAT32_IO, 0x10)
  INIT_FEATURE(bool, VIP_V7, 0x10)
  INIT_FEATURE(bool, TF_QUANTIZATION, 0x10)
  INIT_FEATURE(bool, NN_POST_OUT_SUPPORT_FP32, 0x10)
  INIT_FEATURE(bool, NN_POST_OUT_SUPPORT_FP16, 0x10)
  INIT_FEATURE(bool, NN_POST_OUT_SUPPORT_BF16, 0x10)
  INIT_FEATURE(bool, OUTPUT_CONVERT_UINT8_INT8_TO_UINT16_INT16_FIX, 0x10)
  INIT_FEATURE(bool, NN_SUPPORT_16_8_QUANTIZATION, 0x10)
  INIT_FEATURE(bool, NN_POST_MULT_SUPPORT_FP_CONV, 0x10)

  // tensor processor engine features
  INIT_FEATURE(uint32_t, TPEngine_CoreCount, 0x10)
  INIT_FEATURE(bool, TP_FLOAT32_IO, 0x10)
  INIT_FEATURE(uint32_t, TPLite_CoreCount, 0x10)

  // ppu features
  INIT_FEATURE(uint32_t, NumShaderCores, 0x2)
  INIT_FEATURE(uint32_t, ThreadCount, 0x100)
  INIT_FEATURE(bool, EVIS_VX2, 0x10)
};

#define __PID_NAME__ VSI_VIP_HARDWARE
#include "spec/hw/pid_spec_list.h"
#undef __PID_NAME__
