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
#include <unordered_map>
#include <vector>

#include "NeuralNetworksTypes.h"
#include "OpCreator.h"
#include "Types.h"
#include "tim/vx/tensor.h"
#include "Memory.h"
namespace vsi {
namespace android {
namespace sl {

class Model {
   public:
    Model() : operand_id_(0), relaxed_(false), finished_(false) {}
    int AddOperand(const ANeuralNetworksOperandType& type);
    int SetOperandSymmPerChannelQuantParams(
            int32_t index, const ANeuralNetworksSymmPerChannelQuantParams& channelQuant);
    int SetOperandValue(uint32_t index, const void* buffer, size_t length);
    int SetOperandValueFromMemory(int32_t index, const Memory* memory, size_t offset,
                                  size_t length);
    int AddOperation(ANeuralNetworksOperationType type, uint32_t inputCount, const uint32_t* inputs,
                     uint32_t outputCount, const uint32_t* outputs);
    int IdentifyInputsAndOutputs(uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
                                 const uint32_t* outputs);
    int RelaxComputationFloat32toFloat16(bool allow) {
        if (finished_) {
            std::cout << "can not modify a finished model." << std::endl;
            return ANEURALNETWORKS_BAD_STATE;
        }
        relaxed_ = allow;
        return ANEURALNETWORKS_NO_ERROR;
    }
    int Finish() {
        finished_ = true;
        return ANEURALNETWORKS_NO_ERROR;
    }
    int GetSupportedOperations(bool* supported_ops) const;
    const TensorMap& Tensors() const { return tensors_; }
    const ScalarMap& Scalars() const { return scalars_; }
    TensorMap& Tensors() { return tensors_; }
    ScalarMap& Scalars() { return scalars_; }
    std::vector<std::shared_ptr<OpCreator>>& Operations() { return op_creators_; }

    const std::vector<uint32_t>& Inputs() { return inputs_; }
    const std::vector<uint32_t>& Outputs() { return outputs_; }
    bool IsRelaxed() { return relaxed_; }

   private:
    TensorMap tensors_;
    ScalarMap scalars_;
    std::vector<std::shared_ptr<OpCreator>> op_creators_;
    std::vector<bool> op_supports_;
    std::unordered_map<uint32_t, std::vector<uint8_t>> constant_copy_;
    std::vector<uint32_t> inputs_;
    std::vector<uint32_t> outputs_;
    int32_t operand_id_;
    bool relaxed_;
    bool finished_;
};
}  // namespace sl
}  // namespace android
}  // namespace vsi
#endif