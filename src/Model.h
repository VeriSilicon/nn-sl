/****************************************************************************
 *
 *    Copyright (c) 2022 Vivante Corporation
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

#ifndef VSI_ANDROID_SL_MODEL_H
#define VSI_ANDROID_SL_MODEL_H

#include <android/NeuralNetworksTypes.h>

#include <unordered_map>
#include <vector>

#include "Memory.h"
#include "OpCreator.h"
#include "Types.h"
#include "tim/vx/tensor.h"

namespace vsi::android::sl {

class Model {
   public:
    struct OperandValueInfo {
        size_t size;
        size_t offset;  // Offset in const copy storage.
        const void* buffer;
        const IMemory* memory;
    };
    using OperandValueInfoMap = std::unordered_map<uint32_t, OperandValueInfo>;

    int addOperand(const ANeuralNetworksOperandType& type);
    int setOperandSymmPerChannelQuantParams(
            int32_t index, const ANeuralNetworksSymmPerChannelQuantParams& channelQuant);
    int setOperandValue(int32_t index, const void* buffer, size_t length);
    int setOperandValueFromMemory(int32_t index, const IMemory* memory, size_t offset,
                                  size_t length);
    int setOperandValueFromModel(int32_t index, const Model* reference);
    int addOperation(ANeuralNetworksOperationType type, uint32_t inputCount, const uint32_t* inputs,
                     uint32_t outputCount, const uint32_t* outputs);
    int identifyInputsAndOutputs(uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
                                 const uint32_t* outputs);
    int relaxComputationFloat32toFloat16(bool relaxed);
    int getSupportedOperations(bool* supportedOps) const;
    int finish();

    TensorMap& getTensorMap() { return tensors_; }
    ScalarMap& getScalarMap() { return scalars_; }
    [[nodiscard]] const TensorMap& getTensorMap() const { return tensors_; }
    [[nodiscard]] const ScalarMap& getScalarMap() const { return scalars_; }
    [[nodiscard]] const OperandValueInfoMap& getOperandValueInfos() const {
        return operandValueInfos_;
    }
    [[nodiscard]] const void* getConstantCopyData(size_t offset) const {
        return constantCopyStorage_.data() + offset;
    };
    [[nodiscard]] const std::vector<std::shared_ptr<OpCreator>>& getOpCreators() const {
        return opCreators_;
    }

    std::vector<uint32_t>& getInputs() { return inputs_; }
    std::vector<uint32_t>& getOutputs() { return outputs_; }
    [[nodiscard]] const std::vector<uint32_t>& getInputs() const { return inputs_; }
    [[nodiscard]] const std::vector<uint32_t>& getOutputs() const { return outputs_; }
    [[nodiscard]] bool isRelaxed() const { return relaxed_; }
    [[nodiscard]] bool isFinished() const { return finished_; }

   private:
    TensorMap tensors_;
    ScalarMap scalars_;
    OperandValueInfoMap operandValueInfos_;
    std::vector<const Model*> referenceModels_;
    std::vector<std::shared_ptr<OpCreator>> opCreators_;
    std::vector<bool> opSupported_;
    std::vector<uint8_t> constantCopyStorage_;
    std::vector<uint32_t> inputs_;
    std::vector<uint32_t> outputs_;

    uint32_t numOperands_ = 0;
    bool relaxed_ = false;
    bool finished_ = false;
};

}  // namespace vsi::android::sl
#endif