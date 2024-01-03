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

#include "Device.h"

#include <memory>

#include "NeuralNetworksSupportLibraryImpl.h"
#include "Utils.h"

namespace vsi::android::sl {

Device::Device(std::shared_ptr<tim::vx::platform::IDevice> device) {
    name_ = std::string(kNamePrefix) + '-' + std::to_string(device->Id());
    device_ = std::move(device);
}

Device::PerformanceInfo Device::queryPerformanceInfo(int32_t kind) const {
    switch (kind) {
        case SL_ANEURALNETWORKS_CAPABILITIES_PERFORMANCE_RELAXED_SCALAR:
        case SL_ANEURALNETWORKS_CAPABILITIES_PERFORMANCE_RELAXED_TENSOR:
            return {
                    0.5F,
                    0.5F,
            };
            break;
        case SL_ANEURALNETWORKS_CAPABILITIES_PERFORMANCE_IF:
        case SL_ANEURALNETWORKS_CAPABILITIES_PERFORMANCE_WHILE:
            return {
                    10.0F,
                    10.0F,
            };
            break;
        default:
            LOGW("Device::queryPerformanceInfo passed an invalid performance info code: %d", kind);
            return {};
    }
}

Device::PerformanceInfo Device::queryOperandPerformanceInfo(OperandType operandType) const {
    switch (operandType) {
        case OperandType::TENSOR_FLOAT32:
        case OperandType::TENSOR_FLOAT16:
        case OperandType::TENSOR_INT32:
        case OperandType::TENSOR_BOOL8:
        case OperandType::TENSOR_QUANT8_ASYMM:
        case OperandType::TENSOR_QUANT8_SYMM:
        case OperandType::TENSOR_QUANT8_ASYMM_SIGNED:
        case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
        case OperandType::TENSOR_QUANT16_ASYMM:
        case OperandType::TENSOR_QUANT16_SYMM:
        case OperandType::FLOAT32:
        case OperandType::FLOAT16:
        case OperandType::INT32:
        case OperandType::UINT32:
        case OperandType::BOOL:
            return {
                    0.5F,
                    0.5F,
            };
            break;
        default:
            LOGW("Device::queryOperandPerformanceInfo passed an unsupported op type: %d",
                 static_cast<int>(operandType));
            return {};
    }
}

}  // namespace vsi::android::sl