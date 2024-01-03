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

#include "MemoryDesc.h"

#include <numeric>

#include "Compilation.h"
#include "Model.h"
#include "Utils.h"

namespace vsi::android::sl {

int MemoryDesc::addRole(const Compilation* compilation, IOType ioType, uint32_t index,
                        float frequency) {
    if (finished_) {
        LOGE("MemoryDesc::addRole called after the memory desc is finished");
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (!compilation->isFinished()) {
        LOGE("MemoryDesc::addRole passed an unfinished compilation");
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (frequency <= 0.0F || frequency > 1.0F) {
        LOGE("MemoryDesc::addRole passed an invalid frequency");
        return ANEURALNETWORKS_BAD_DATA;
    }

    if (roles_.count({compilation, ioType, index}) > 0) {
        LOGE("MemoryDesc::addRole the same role is specified twice");
        return ANEURALNETWORKS_BAD_DATA;
    }

    const auto* model = compilation->getModel();
    const auto& tensorMap = model->getTensorMap();
    slang::type::tensor_storage tensorOperand;

    if (ioType == IOType::INPUT) {
        const auto& inputs = model->getInputs();
        if (index >= inputs.size()) {
            LOGE("MemoryDesc::addRole input index (%u) out of range", index);
            return ANEURALNETWORKS_BAD_DATA;
        }

        uint32_t input = inputs[index];
        if (tensorMap.count(input) == 0) {
            LOGE("MemoryDesc::addRole cannot find corresponding tensor for input index (%u)",
                 index);
            return ANEURALNETWORKS_BAD_DATA;
        }

        tensorOperand = tensorMap.at(input);
    } else if (ioType == IOType::OUTPUT) {
        const auto& outputs = model->getOutputs();
        if (index >= outputs.size()) {
            LOGE("MemoryDesc::addRole output index (%u) out of range", index);
            return ANEURALNETWORKS_BAD_DATA;
        }

        uint32_t output = outputs[index];
        if (tensorMap.count(output) == 0) {
            LOGE("MemoryDesc::addRole cannot find corresponding tensor for output index (%u)",
                 index);
            return ANEURALNETWORKS_BAD_DATA;
        }

        tensorOperand = tensorMap.at(output);
    }

    if (tensorOperand_.has_value()) {
        if (tensorOperand.attr != tensorOperand_->attr ||
            tensorOperand.dtype != tensorOperand_->dtype ||
            tensorOperand.scale != tensorOperand_->scale ||
            tensorOperand.zero_point != tensorOperand_->zero_point ||
            tensorOperand.per_channel_scales != tensorOperand_->per_channel_scales) {
            LOGE("MemoryDesc::addRole incompatible tensor metadata");
            return ANEURALNETWORKS_BAD_DATA;
        }
    } else {
        tensorOperand_ = tensorOperand;
    }

    auto shape = combineShape(shape_, tensorOperand.shape);
    if (shape.empty()) {
        LOGE("MemoryDesc::addRole incompatible tensor shapes");
        return ANEURALNETWORKS_BAD_DATA;
    }
    shape_ = shape;

    roles_.insert({compilation, ioType, index});
    return ANEURALNETWORKS_NO_ERROR;
}

int MemoryDesc::setShape(const std::vector<uint32_t>& dimensions) {
    if (finished_) {
        LOGE("MemoryDesc::setDimensions called after the memory desc is finished");
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (tensorOperand_.has_value() && dimensions.empty()) {
        LOGE("MemoryDesc::setDimensions incompatible shapes for scalars");
        return ANEURALNETWORKS_BAD_DATA;
    }

    auto shape = combineShape(shape_, dimensions);
    if (shape.empty() && !dimensions.empty()) {
        LOGE("MemoryDesc::setDimensions incompatible shapes");
        return ANEURALNETWORKS_BAD_DATA;
    }

    shape_ = shape;
    return ANEURALNETWORKS_NO_ERROR;
}

int MemoryDesc::finish() {
    if (finished_) {
        LOGE("MemoryDesc::finish called after the memory desc is finished");
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (roles_.empty()) {
        LOGE("MemoryDesc::finish the memory desc has no role");
        return ANEURALNETWORKS_BAD_STATE;
    }

    for (auto [c0, t0, i0] : roles_) {
        for (auto [c1, t1, i1] : roles_) {
            if (c0 == c1 && t0 != t1) {
                LOGE("MemoryDesc::finish the same device memory cannot be used for both input and "
                     "output of the same compilation");
                return ANEURALNETWORKS_BAD_STATE;
            }
        }
    }

    finished_ = true;
    return ANEURALNETWORKS_NO_ERROR;
}

size_t MemoryDesc::getSize() const {
    if (!finished_) {
        return 0;
    }

    size_t numElements = std::reduce(shape_.cbegin(), shape_.cend(), 1, std::multiplies<size_t>());
    size_t elementSize = getDtypeSize(tensorOperand_->dtype);
    return elementSize * numElements;
}

}  // namespace vsi::android::sl