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
#ifndef VSI_ANDROID_SL_MAP_OPERATION_H
#define VSI_ANDROID_SL_MAP_OPERATION_H

#include "Model.h"
#include "Types.h"
#include "tim/vx/graph.h"

namespace vsi {
namespace android {
namespace sl {

int MapOneInputOneOutput(std::shared_ptr<tim::vx::Graph> graph,
                         std::shared_ptr<OpCreator> op_creator,
                         std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                         const TensorMap& tensor_map, const ScalarMap& scalar_map,
                         const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapBatchMatmul(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                   std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                   const TensorMap& tensor_map, const ScalarMap& scalar_map,
                   const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapConcatenation(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                     std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                     const TensorMap& tensor_map, const ScalarMap& scalar_map,
                     const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapConv2D(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
              std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
              const TensorMap& tensor_map, const ScalarMap& scalar_map,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapDepthwiseConv2D(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                       std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                       const TensorMap& tensor_map, const ScalarMap& scalar_map,
                       const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapEltwise(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
               std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
               const TensorMap& tensor_map, const ScalarMap& scalar_map,
               const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapEltwiseWithNoAct(std::shared_ptr<tim::vx::Graph> graph,
                        std::shared_ptr<OpCreator> op_creator,
                        std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                        const TensorMap& tensor_map, const ScalarMap& scalar_map,
                        const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapEmbeddingLookup(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                       std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                       const TensorMap& tensor_map, const ScalarMap& scalar_map,
                       const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapGather(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
              std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
              const TensorMap& tensor_map, const ScalarMap& scalar_map,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapFullyConnected(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                      std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                      const TensorMap& tensor_map, const ScalarMap& scalar_map,
                      const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapGroupedConv2d(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                     std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                     const TensorMap& tensor_map, const ScalarMap& scalar_map,
                     const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapHashtableLookup(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                       std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                       const TensorMap& tensor_map, const ScalarMap& scalar_map,
                       const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapInstanceNormalization(
        std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
        std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
        const TensorMap& tensor_map, const ScalarMap& scalar_map,
        const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapLogicalAndOr(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                    std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                    const TensorMap& tensor_map, const ScalarMap& scalar_map,
                    const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapPack(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
            std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
            const TensorMap& tensor_map, const ScalarMap& scalar_map,
            const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapPool2D(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
              std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
              const TensorMap& tensor_map, const ScalarMap& scalar_map,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapPrelu(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
             std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
             const TensorMap& tensor_map, const ScalarMap& scalar_map,
             const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapRelationalOp(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                    std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                    const TensorMap& tensor_map, const ScalarMap& scalar_map,
                    const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapRoi(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                const TensorMap& tensor_map, const ScalarMap& scalar_map,
                const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapSelect(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
              std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
              const TensorMap& tensor_map, const ScalarMap& scalar_map,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapSplit(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
             std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
             const TensorMap& tensor_map, const ScalarMap& scalar_map,
             const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapSvdf(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
            std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
            const TensorMap& tensor_map, const ScalarMap& scalar_map,
            const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapTopK(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
            std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
            const TensorMap& tensor_map, const ScalarMap& scalar_map,
            const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapTranspose(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                 std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                 const TensorMap& tensor_map, const ScalarMap& scalar_map,
                 const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int MapTransposeConv2d(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                       std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                       const TensorMap& tensor_map, const ScalarMap& scalar_map,
                       const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
}  // namespace sl
}  // namespace android
}  // namespace vsi

#endif