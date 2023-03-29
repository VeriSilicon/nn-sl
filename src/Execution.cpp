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
#include "Execution.h"

#include <algorithm>

#include "MapOperation.h"
#include "Memory.h"
#include "Utils.h"
#include "tim/transform/layout_inference.h"
#include "tim/vx/ops.h"
#include "tim/vx/platform/native.h"
#include "tim/vx/platform/platform.h"
#include "tim/vx/tensor.h"

namespace vsi {
namespace android {
namespace sl {

std::shared_ptr<tim::vx::Tensor> Execution::CreateTvxIOTensor(
        const slang::type::tensor_storage& tensor, tim::vx::TensorAttribute attr) {
    tim::vx::DataType data_type = ToTvxDataType(tensor.dtype);
    tim::vx::ShapeType shape = tensor.shape;
    std::reverse(shape.begin(), shape.end());
    tim::vx::Quantization quantization;
    tim::vx::QuantType quant_type = ToTvxQuantType(tensor.qtype);
    if (quant_type == tim::vx::QuantType::ASYMMETRIC) {
        quantization = tim::vx::Quantization(quant_type, tensor.scale, tensor.zero_point);
    } else if (quant_type == tim::vx::QuantType::SYMMETRIC_PER_CHANNEL) {
        quantization =
                tim::vx::Quantization(quant_type, tensor.channel_dim, tensor.per_channel_scales,
                                      tensor.per_channel_zero_points);
    }
    tim::vx::TensorSpec spec(data_type, shape, attr, quantization);
    return vx_graph_->CreateIOTensor(spec);
}

int Execution::SetInput(int32_t index, const ANeuralNetworksOperandType* type, const void* buffer,
                        size_t length) {
    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::MapOperations(const std::vector<std::shared_ptr<OpCreator>>& op_creators,
                             const TensorMap& tensor_map, const ScalarMap& scalar_map) {
    for (const auto op_creator : op_creators) {
        const std::vector<uint32_t>& inputs = op_creator->Inputs();
        const std::vector<uint32_t>& outputs = op_creator->Outputs();
        int result = ANEURALNETWORKS_NO_ERROR;
        switch (op_creator->Type()) {
            case ANEURALNETWORKS_ABS:
                result = MapEltwiseUnary(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_ADD:
                result = MapEltwise(vx_graph_, op_creator, vx_tensors_, tensor_map, scalar_map, inputs,
                                outputs);
                break;
            case ANEURALNETWORKS_AVERAGE_POOL_2D:
                result = MapPool2D(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                          scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_BATCH_TO_SPACE_ND:
                result = MapBatchToSpace(vx_graph_, op_creator, vx_tensors_, tensor_map, scalar_map,
                                   inputs, outputs);
                break;
            case ANEURALNETWORKS_CONV_2D:
                result = MapConv2D(vx_graph_, op_creator, vx_tensors_, tensor_map, scalar_map,
                                   inputs, outputs);
                break;
            case ANEURALNETWORKS_DEPTHWISE_CONV_2D:
                result = MapDepthwiseConv2D(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_DEPTH_TO_SPACE:
                result = MapDepthToSpace(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_DEQUANTIZE:
                result = MapDataConvert(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_DIV:
                result = MapEltwise(vx_graph_, op_creator, vx_tensors_, tensor_map, scalar_map, inputs,
                                outputs);
                break;
            case ANEURALNETWORKS_ELU:
                result = MapActivation(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_EQUAL:
                result = MapRelationalOp(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_EXP:
                result = MapEltwiseUnary(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_FLOOR:
                result = MapEltwiseUnary(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_FULLY_CONNECTED:
                result = MapFullyConnected(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_GREATER:
                result = MapRelationalOp(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_GREATER_EQUAL:
                result = MapRelationalOp(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_GROUPED_CONV_2D:
                result = MapGroupedConv2d(vx_graph_, op_creator, vx_tensors_, tensor_map, scalar_map,
                                   inputs, outputs);
                break;
            case ANEURALNETWORKS_HARD_SWISH:
                result = MapActivation(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_L2_NORMALIZATION:
                result = MapL2Normalization(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_LESS:
                result = MapRelationalOp(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_LESS_EQUAL:
                result = MapRelationalOp(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_LOG:
                result = MapEltwiseUnary(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_LOGISTIC:
                result = MapActivation(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_LOGICAL_AND:
                result = MapLogicalAndOr(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_LOGICAL_NOT:
                result = MapLogcialNot(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_LOGICAL_OR:
                result = MapLogicalAndOr(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_L2_POOL_2D:
                result = MapPool2D(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_MAX_POOL_2D:
                result = MapPool2D(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_MEAN:
                result = MapMean(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_MUL:
                result = MapEltwise(vx_graph_, op_creator, vx_tensors_, tensor_map, scalar_map, inputs,
                                outputs);
                break;
            case ANEURALNETWORKS_NEG:
                result = MapEltwiseUnary(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_NOT_EQUAL:
                result = MapRelationalOp(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_PAD:
                result = MapPad(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_PAD_V2:
                result = MapPadV2(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_POW:
                result = MapPow(vx_graph_, op_creator, vx_tensors_, tensor_map, scalar_map, inputs,
                                outputs);
                break;
            case ANEURALNETWORKS_PRELU:
                result = MapPrelu(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_QUANTIZE:
                result = MapDataConvert(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_REDUCE_ALL:
                result = MapReduce(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_REDUCE_ANY:
                result = MapReduce(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_REDUCE_MAX:
                result = MapReduce(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_REDUCE_MIN:
                result = MapReduce(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_REDUCE_PROD:
                result = MapReduce(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_REDUCE_SUM:
                result = MapReduce(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_RELU:
                result = MapActivation(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_RELU1:
                result = MapActivation(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_RELU6:
                result = MapActivation(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_RESHAPE:
                result = MapReshape(vx_graph_, op_creator, vx_tensors_, tensor_map, scalar_map,
                                    inputs, outputs);
                break;
            case ANEURALNETWORKS_RSQRT:
                result = MapEltwiseUnary(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_SIN:
                result = MapEltwiseUnary(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_SOFTMAX:
                result = MapSoftmax(vx_graph_, op_creator, vx_tensors_, tensor_map, scalar_map,
                                    inputs, outputs);
                break;
            case ANEURALNETWORKS_SPACE_TO_DEPTH:
                result = MapSpaceToDepth(vx_graph_, op_creator, vx_tensors_, tensor_map, scalar_map,
                                    inputs, outputs);
                break;
            case ANEURALNETWORKS_SPACE_TO_BATCH_ND:
                result = MapSpaceToBatch(vx_graph_, op_creator, vx_tensors_, tensor_map, scalar_map,
                                    inputs, outputs);
                break;
            case ANEURALNETWORKS_SQRT:
                result = MapEltwiseUnary(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_SUB:
                result = MapEltwise(vx_graph_, op_creator, vx_tensors_, tensor_map, scalar_map, inputs,
                                outputs);
                break;
            case ANEURALNETWORKS_TANH:
                result = MapActivation(vx_graph_, op_creator, vx_tensors_, tensor_map,
                                            scalar_map, inputs, outputs);
                break;
            case ANEURALNETWORKS_TRANSPOSE_CONV_2D:
                result = MapTransposeConv2d(vx_graph_, op_creator, vx_tensors_, tensor_map, scalar_map,
                                   inputs, outputs);
                break;
            default:
                std::cout << "Op type: " << op_creator->Type() << " is not supported" << std::endl;
                result = ANEURALNETWORKS_BAD_STATE;
        }
        if (result != ANEURALNETWORKS_NO_ERROR) return result;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::SetInputFromMemory(int32_t index, const ANeuralNetworksOperandType* type,
                                  const Memory* memory, size_t offset, size_t length) {
    if (type != nullptr) {
        Model* model = compilation_->GetModel();
        int32_t input = model->Inputs()[index];
        auto& tensors = model->Tensors();
        auto& input_tensor = tensors[input];
        if (input_tensor.dtype != MapDataType(type->type) || input_tensor.scale != type->scale ||
            input_tensor.zero_point != type->zeroPoint) {
            std::cout << "Get invalid ANeuralNetworksOperandType when setting input." << std::endl;
            return ANEURALNETWORKS_BAD_DATA;
        }
        inputs_dimension_[index] =
                std::vector<uint32_t>(type->dimensions, type->dimensions + type->dimensionCount);
        input_tensor.shape = inputs_dimension_[index];
    }
    inputs_memory_[index] = IOMemory(memory, offset, length);

    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::SetOutputFromMemory(int32_t index, const ANeuralNetworksOperandType* type,
                                   const Memory* memory, size_t offset, size_t length) {
    if (type != nullptr) {
        Model* model = compilation_->GetModel();
        int32_t output = model->Outputs()[index];
        auto& tensors = model->Tensors();
        auto& output_tensor = tensors[output];
        if (output_tensor.dtype != MapDataType(type->type) || output_tensor.scale != type->scale ||
            output_tensor.zero_point != type->zeroPoint) {
            std::cout << "Get invalid ANeuralNetworksOperandType when setting output." << std::endl;
            return ANEURALNETWORKS_BAD_DATA;
        }
        outputs_dimension_[index] =
                std::vector<uint32_t>(type->dimensions, type->dimensions + type->dimensionCount);
        output_tensor.shape = outputs_dimension_[index];
    }
    outputs_memory_[index] = IOMemory(memory, offset, length);
    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::Compute() {
    Model* model = compilation_->GetModel();
    if (vx_graph_ == nullptr) {
        // construct graph
        vx_graph_ = vx_context_->CreateGraph();
        auto tensor_map = model->Tensors();
        auto scalar_map = model->Scalars();
        auto operations = model->Operations();
        for (uint32_t in : model->Inputs()) {
            vx_tensors_[in] = CreateTvxIOTensor(tensor_map[in], tim::vx::TensorAttribute::INPUT);
        }
        for (uint32_t out : model->Outputs()) {
            vx_tensors_[out] = CreateTvxIOTensor(tensor_map[out], tim::vx::TensorAttribute::OUTPUT);
        }

        int result = MapOperations(operations, tensor_map, scalar_map);
        if (result != ANEURALNETWORKS_NO_ERROR) {
            std::cout << "map operation fail" << std::endl;
            return result;
        }

        layout_infered_ = tim::transform::LayoutInference(vx_graph_, vx_context_);
#ifdef RUN_NBG
        // compile graph to executable, just use the first device
        auto device = compilation_->Devices()[0]->Device();
        executor_ = std::make_shared<tim::vx::platform::NativeExecutor>(device);
        executable_ = executor_->Compile(layout_infered_.first);
        input_handles_.clear();
        output_handles_.clear();
        for (uint32_t i : model->Inputs()) {
            auto input_handle = executable_->AllocateTensor(vx_tensors_[i]->GetSpec());
            executable_->SetInput(input_handle);
            input_handles_.push_back(input_handle);
        }
        for (uint32_t o : model->Outputs()) {
            auto output_handle = executable_->AllocateTensor(vx_tensors_[o]->GetSpec());
            executable_->SetOutput(output_handle);
            output_handles_.push_back(output_handle);
        }
#endif

        auto inputs = model->Inputs();
        for (int i = 0; i < inputs.size(); ++i) {
#ifdef RUN_NBG
            auto input_handle = input_handles_[i];
            auto io_memory = inputs_memory_[i];
#else
            uint32_t index = inputs[i];
            auto src_input_tensor = vx_tensors_[index];
            auto io_memory = inputs_memory_[i];
#endif
            auto memory = io_memory.memory;
            size_t offset = io_memory.offset;
            size_t length = io_memory.length;
            if (offset + length > memory->Length()) {
                std::cout << "input memory is out of range." << std::endl;
                return ANEURALNETWORKS_OUT_OF_MEMORY;
            }
            uint8_t* data = reinterpret_cast<uint8_t*>(memory->Data());
#ifdef RUN_NBG
            if (!input_handle->CopyDataToTensor(reinterpret_cast<void*>(data + offset), length)) {
                std::cout << "copy data to tensor fail." << std::endl;
                return ANEURALNETWORKS_BAD_STATE;
            }
#else
            auto infered_input_tensor = layout_infered_.second[src_input_tensor];
            if (infered_input_tensor) {
                if (!infered_input_tensor->CopyDataToTensor(reinterpret_cast<void*>(data + offset),
                                                            length)) {
                    std::cout << "copy data to tensor fail." << std::endl;
                    return ANEURALNETWORKS_BAD_STATE;
                }
            } else {
                std::cout << "tensor in source graph removed before do layout "
                             "inference - if zero sized tensor involved"
                          << std::endl;
            }
#endif
        }
#ifdef RUN_NBG
        executable_->Submit(executable_);
#endif
    } else if (reusable_ == false) {
        std::cout << "try to schedule multiple computations for a Execution which is not reusable"
                  << std::endl;
        return ANEURALNETWORKS_BAD_STATE;
    }

#ifdef RUN_NBG
    executor_->Trigger();
#else
    // Run graph
    if (!layout_infered_.first->Run()) {
        std::cout << "failed to run graph." << std::endl;
        return ANEURALNETWORKS_BAD_STATE;
    }
#endif
    // copy output to memory
    auto outputs = model->Outputs();
    for (int i = 0; i < outputs.size(); ++i) {
#ifdef RUN_NBG
        auto output_handle = output_handles_[i];
        auto io_memory = outputs_memory_[i];
#else
        uint32_t index = outputs[i];
        auto src_output_tensor = vx_tensors_[index];
        auto io_memory = outputs_memory_[i];
#endif
        auto memory = io_memory.memory;
        size_t offset = io_memory.offset;
        size_t length = io_memory.length;
        if (offset + length > memory->Length()) {
            std::cout << "output memory is out of range." << std::endl;
            return ANEURALNETWORKS_BAD_DATA;
        }
        uint8_t* data = reinterpret_cast<uint8_t*>(memory->Data());
#ifdef RUN_NBG
        if (!output_handle->CopyDataFromTensor(reinterpret_cast<void*>(data + offset))) {
            std::cout << "copy data from tensor fail." << std::endl;
            return ANEURALNETWORKS_BAD_STATE;
        }
#else
        auto infered_output_tesnor = layout_infered_.second[src_output_tensor];
        if (infered_output_tesnor) {
            if (!infered_output_tesnor->CopyDataFromTensor(
                        reinterpret_cast<void*>(data + offset))) {
                std::cout << "copy data from tensor fail." << std::endl;
                return ANEURALNETWORKS_BAD_STATE;
            }
        } else {
            std::cout << "Output tensor missing: report issue to VSI" << std::endl;
        }
#endif
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::GetOutputOperandRank(int32_t index, uint32_t* rank) {
    *rank = outputs_dimension_[index].size();
    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::GetOutputOperandDimensions(int32_t index, uint32_t* dimensions) {
    auto dim = outputs_dimension_[index];
    for (int i = 0; i < dim.size(); ++i) {
        dimensions[i] = dim[i];
    }
    return ANEURALNETWORKS_NO_ERROR;
}

}  // namespace sl
}  // namespace android
}  // namespace vsi