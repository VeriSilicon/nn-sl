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

namespace vsi::android::sl {

int mapOneInputOneOutput(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                         const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                         const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                         const std::vector<uint32_t>& outputs);
int mapBatchMatmul(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                   const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                   const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                   const std::vector<uint32_t>& outputs);
int mapConcatenation(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                     const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                     const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                     const std::vector<uint32_t>& outputs);
int mapConv2D(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
              const TensorMap& tensorMap, const ScalarMap& scalarMap,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int mapDepthwiseConv2D(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                       const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                       const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                       const std::vector<uint32_t>& outputs);
int mapEltwise(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
               const TensorMap& tensorMap, const ScalarMap& scalarMap,
               const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int mapEltwiseWithNoAct(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                        const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                        const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                        const std::vector<uint32_t>& outputs);
int mapEmbeddingLookup(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                       const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                       const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                       const std::vector<uint32_t>& outputs);
int mapGather(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
              const TensorMap& tensorMap, const ScalarMap& scalarMap,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int mapFullyConnected(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                      const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                      const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                      const std::vector<uint32_t>& outputs);
int mapGroupedConv2d(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                     const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                     const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                     const std::vector<uint32_t>& outputs);
int mapHashtableLookup(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                       const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                       const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                       const std::vector<uint32_t>& outputs);
int mapInstanceNormalization(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                             const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                             const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                             const std::vector<uint32_t>& outputs);
int mapLogicalAndOr(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                    const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                    const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                    const std::vector<uint32_t>& outputs);
int mapPack(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
            const TensorMap& tensorMap, const ScalarMap& scalarMap,
            const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int mapPool2D(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
              const TensorMap& tensorMap, const ScalarMap& scalarMap,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int mapPrelu(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
             const TensorMap& tensorMap, const ScalarMap& scalarMap,
             const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int mapRelationalOp(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                    const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                    const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                    const std::vector<uint32_t>& outputs);
int mapRoi(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
           const TensorMap& tensorMap, const ScalarMap& scalarMap,
           const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int mapSelect(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
              const TensorMap& tensorMap, const ScalarMap& scalarMap,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int mapSplit(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
             const TensorMap& tensorMap, const ScalarMap& scalarMap,
             const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int mapSvdf(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
            const TensorMap& tensorMap, const ScalarMap& scalarMap,
            const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int mapTopK(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
            const TensorMap& tensorMap, const ScalarMap& scalarMap,
            const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int mapTranspose(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
                 const TensorMap& tensorMap, const ScalarMap& scalarMap,
                 const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs);
int mapTransposeConv2d(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                       const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                       const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                       const std::vector<uint32_t>& outputs);
}  // namespace vsi::android::sl

#endif