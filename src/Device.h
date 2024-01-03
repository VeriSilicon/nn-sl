/****************************************************************************
 *
 *    Copyright (c) 2024 Vivante Corporation
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#ifndef VSI_ANDROID_SL_VSI_DEVICE_H
#define VSI_ANDROID_SL_VSI_DEVICE_H

#include <android/NeuralNetworksTypes.h>

#include <string>
#include <string_view>

#include "Types.h"
#include "tim/vx/platform/platform.h"

namespace vsi::android::sl {

class Device {
    struct PerformanceInfo {
        float execTimeRatio;
        float powerUsageRatio;
    };

   public:
    static constexpr std::array<OperandType, 15> kSupportedOperandTypes = {
            OperandType::FLOAT32,
            OperandType::INT32,
            OperandType::UINT32,
            OperandType::TENSOR_FLOAT32,
            OperandType::TENSOR_INT32,
            OperandType::TENSOR_QUANT8_ASYMM,
            OperandType::BOOL,
            OperandType::TENSOR_QUANT16_SYMM,
            OperandType::TENSOR_FLOAT16,
            OperandType::TENSOR_BOOL8,
            OperandType::FLOAT16,
            OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL,
            OperandType::TENSOR_QUANT16_ASYMM,
            OperandType::TENSOR_QUANT8_SYMM,
            OperandType::TENSOR_QUANT8_ASYMM_SIGNED,
    };

    explicit Device(std::shared_ptr<tim::vx::platform::IDevice> device);

    [[nodiscard]] std::string_view getName() const { return name_; }
    [[nodiscard]] std::string_view getVersion() const { return kVersion; }
    [[nodiscard]] int64_t getFeatureLevel() const { return kFeatureLevel; }
    [[nodiscard]] PerformanceInfo queryPerformanceInfo(int32_t kind) const;
    [[nodiscard]] PerformanceInfo queryOperandPerformanceInfo(OperandType operandType) const;

   private:
    std::string name_;
    std::shared_ptr<tim::vx::platform::IDevice> device_;

    static constexpr std::string_view kNamePrefix = "vsi-device";
    static constexpr std::string_view kVersion = "0.0.1";
    static constexpr int64_t kFeatureLevel = ANEURALNETWORKS_FEATURE_LEVEL_7;
};

}  // namespace vsi::android::sl

#endif