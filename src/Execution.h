/****************************************************************************
 *
 *    copyright (c) 2023 Vivante Corporation
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
#include <map>

#include "Compilation.h"
#include "Memory.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"

namespace vsi {
namespace android {
namespace sl {

class Execution {
   public:
    Execution() {}
    Execution(Compilation* compilation) : compilation_(compilation) {
        auto model = compilation_->GetModel();
        inputs_memory_.resize(model->Inputs().size());
        outputs_memory_.resize(model->Outputs().size());
        inputs_dimension_.resize(model->Inputs().size());
        outputs_dimension_.resize(model->Outputs().size());
        vx_context_ = tim::vx::Context::Create();
        reusable_ = false;
    }

    int SetReusable(bool reusable) {
        reusable_ = reusable;
        return ANEURALNETWORKS_NO_ERROR;
    }

    int SetInput(int32_t index, const ANeuralNetworksOperandType* type, const void* buffer,
                 size_t length);
    int SetInputFromMemory(int32_t index, const ANeuralNetworksOperandType* type,
                           const Memory* memory, size_t offset, size_t length);
    int SetOutputFromMemory(int32_t index, const ANeuralNetworksOperandType* type,
                            const Memory* memory, size_t offset, size_t length);
    int Compute();
    int GetOutputOperandRank(int32_t index, uint32_t* rank);
    int GetOutputOperandDimensions(int32_t index, uint32_t* dimensions);

   private:
    std::shared_ptr<tim::vx::Tensor> CreateTvxIOTensor(const slang::type::tensor_storage& tensor,
                                                       tim::vx::TensorAttribute attr);
    int MapOperations(const std::vector<std::shared_ptr<OpCreator>>& op_creators,
                      const TensorMap& tensor_map, const ScalarMap& scalar_map);
    struct IOMemory {
        IOMemory() {}
        IOMemory(const Memory* memory, size_t offset, size_t length)
            : memory(memory), offset(offset), length(length) {}
        const Memory* memory;
        size_t offset;
        size_t length;
    };

    std::vector<IOMemory> inputs_memory_;
    std::vector<IOMemory> outputs_memory_;
    std::vector<std::vector<uint32_t>> inputs_dimension_;
    std::vector<std::vector<uint32_t>> outputs_dimension_;
    Compilation* compilation_;
    std::shared_ptr<tim::vx::Context> vx_context_;
    std::shared_ptr<tim::vx::Graph> vx_graph_;
    std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>> vx_tensors_;
    std::pair<std::shared_ptr<tim::vx::Graph>,
              std::map<std::shared_ptr<tim::vx::Tensor>, std::shared_ptr<tim::vx::Tensor>>>
            layout_infered_;
    bool reusable_;

#ifdef RUN_NBG
    std::shared_ptr<tim::vx::platform::IExecutor> executor_;
    std::shared_ptr<tim::vx::platform::IExecutable> executable_;
    std::vector<std::shared_ptr<tim::vx::platform::ITensorHandle>> input_handles_;
    std::vector<std::shared_ptr<tim::vx::platform::ITensorHandle>> output_handles_;
#endif
};

}  // namespace sl
}  // namespace android
}  // namespace vsi

#endif