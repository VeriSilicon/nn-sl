/****************************************************************************
 *
 *    Copyright (c) 2023 Vivante Corporation
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
#ifndef VSI_ANDROID_SL_TYPE_H
#define VSI_ANDROID_SL_TYPE_H

#include <vector>

#include "NeuralNetworksTypes.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"

namespace vsi {
namespace android {
namespace sl {

struct Operand {
    Operand(ANeuralNetworksOperandType type) : type_info(type) {}

    ANeuralNetworksOperandType type_info;

    // symmetric per-channel quantized parameters
    std::vector<float> scales;
    uint32_t channel_dim;

    bool is_small_value{false};
    // store small value by copy
    std::vector<uint8_t> small_value;
    // sotre large value by reference
    const void* buffer{nullptr};
    size_t length{0};
};

struct Operation {
    ANeuralNetworksOperationType type;
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> outputs;
};

enum class MemoryType { FD, DESC, AHB };

}  // namespace sl
}  // namespace android
}  // namespace vsi

#endif