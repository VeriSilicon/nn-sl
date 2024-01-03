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

#ifndef VSI_ANDROID_SL_EXECUTION_H
#define VSI_ANDROID_SL_EXECUTION_H

#include <memory>
#include <unordered_map>
#include <vector>

#include "Compilation.h"
#include "Event.h"
#include "Memory.h"
#include "Types.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"

namespace vsi::android::sl {

class Execution {
   public:
    explicit Execution(Compilation* compilation);

    [[nodiscard]] const Compilation* getCompilation() const { return compilation_; }

    int setReusable(bool reusable);
    int setTimeout(Duration duration);
    int setLoopTimeout(Duration duration);
    int setMeasureTiming(bool measure);

    int setInput(int32_t index, const ANeuralNetworksOperandType* type, const void* buffer,
                 size_t length);
    int setOutput(int32_t index, const ANeuralNetworksOperandType* type, void* buffer,
                  size_t length);
    int setInputFromMemory(int32_t index, const ANeuralNetworksOperandType* type,
                           const IMemory* memory, size_t offset, size_t length);
    int setOutputFromMemory(int32_t index, const ANeuralNetworksOperandType* type,
                            const IMemory* memory, size_t offset, size_t length);

    int compute();

    CallbackEvent* createSyncEvent();
    int startCompute();

    int getDuration(DurationCode durationCode, uint64_t* duration) const;
    int getOutputOperandRank(int32_t index, uint32_t* rank) const;
    int getOutputOperandDimensions(int32_t index, uint32_t* dimensions) const;

   private:
    // See execution state definitions in
    // https://developer.android.com/ndk/reference/group/neural-networks#aneuralnetworksexecution
    enum class State {
        PREPARATION,
        COMPUTATION,
        COMPLETED,
    };

    struct IOBufferInfo {
        size_t offset;
        size_t length;
        void* buffer;
        const IMemory* memory;
    };

    using VxContext = std::shared_ptr<tim::vx::Context>;
    using VxGraph = std::shared_ptr<tim::vx::Graph>;
    using VxTensor = std::shared_ptr<tim::vx::Tensor>;
    using VxOp = std::shared_ptr<tim::vx::Operation>;
    using VxTensorMap = std::unordered_map<uint32_t, VxTensor>;

    VxTensor createVxConstantTensor(const slang::type::tensor_storage& tensor,
                                    Model::OperandValueInfo valueInfo);
    VxTensor createVxIOTensor(const slang::type::tensor_storage& tensor,
                              tim::vx::TensorAttribute attr);
    int mapOperations(const std::vector<std::shared_ptr<OpCreator>>& opCreators,
                      const TensorMap& tensorMap, const ScalarMap& scalarMap);
    int compile();

    // Indexed by execution I/O index, not model tensor index.
    std::vector<IOBufferInfo> inputBufferInfos_;
    std::vector<IOBufferInfo> outputBufferInfos_;
    std::vector<VxTensor> inputVxTensors_;
    std::vector<VxTensor> outputVxTensors_;

    Compilation* compilation_;
    // Compile time graph.
    VxGraph vxGraph_;
    // Runtime graph.
    VxGraph runtimeGraph_;
    // Indexed by model tensor index.
    VxTensorMap vxTensors_;

    CallbackEvent* syncEvent_;
    Duration timeoutDuration_;
    Duration loopTimeoutDuration_;

    bool reusable_;
    bool measure_;
    State state_;
};

}  // namespace vsi::android::sl

#endif