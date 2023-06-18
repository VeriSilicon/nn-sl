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
#include "Model.h"

#include <cassert>
#include <cstring>

#include "Memory.h"
#include "Utils.h"

namespace vsi {
namespace android {
namespace sl {

int Model::AddOperand(const ANeuralNetworksOperandType& type) {
    if (finished_) {
        std::cout << "Error: can not modify a finished model." << std::endl;
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (type.dimensionCount) {  // implies tensor
        if (type.dimensions == nullptr) {
            std::cout << "Error: get an invalid operand" << std::endl;
            return ANEURALNETWORKS_BAD_DATA;
        }
        slang::type::tensor_storage tensor = {
                .dtype = MapDataType(type.type),
                .qtype = MapQuantType(type.type),
                .shape = std::vector<uint32_t>(type.dimensions,
                                               type.dimensions + type.dimensionCount),
                .scale = type.scale,
                .zero_point = type.zeroPoint};
        tensors_.insert({operand_id_++, tensor});
    } else {  // implies scalar
        if (type.dimensions != nullptr) {
            std::cout << "Error: get an invalid operand" << std::endl;
            return ANEURALNETWORKS_BAD_DATA;
        }
        slang::type::scalar_storage scalar = {.dtype = MapDataType(type.type)};
        scalars_.insert({operand_id_++, scalar});
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int Model::SetOperandSymmPerChannelQuantParams(
        int32_t index, const ANeuralNetworksSymmPerChannelQuantParams& channelQuant) {
    if (finished_) {
        std::cout << "Error: can not modify a finished model." << std::endl;
        return ANEURALNETWORKS_BAD_STATE;
    }
    if (index >= operand_id_) {
        std::cout << "ANeuralNetworksModel_SetOperandSymmPerChannelQuantParams get an invalid index"
                  << std::endl;
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (tensors_.find(index) != tensors_.end()) {
        // reverse channel_dim axis
        uint32_t channel_dim = tensors_[index].shape.size() - channelQuant.channelDim - 1;
        tensors_[index].channel_dim = channel_dim;
        tensors_[index].per_channel_scales.assign(channelQuant.scales,
                                                  channelQuant.scales + channelQuant.scaleCount);
        tensors_[index].per_channel_zero_points.assign(channelQuant.scaleCount, 0);
    } else {
        std::cout << "Error: Invalid operand index." << std::endl;
        return ANEURALNETWORKS_BAD_DATA;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int Model::SetOperandValue(uint32_t index, const void* buffer, size_t length) {
    if (finished_) {
        std::cout << "Error: can not modify a finished model." << std::endl;
        return ANEURALNETWORKS_BAD_STATE;
    }
    if (index >= operand_id_) {
        std::cout << "ANeuralNetworksModel_setOperandValue get an invalid index" << std::endl;
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (length > 0xFFFFFFFF) {
        std::cout << "ANeuralNetworksModel_setOperandValue value length of " << length
                  << " exceeds max size" << std::endl;
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (buffer == nullptr) {
        std::cout << "Warning: This tensor is empty" << std::endl;
        return ANEURALNETWORKS_NO_ERROR;
    }
    if (length <= ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES) {
        const uint8_t* copied_values = reinterpret_cast<const uint8_t*>(buffer);
        constant_copy_.insert({index, std::vector<uint8_t>(copied_values, copied_values + length)});
    }

    if (tensors_.find(index) != tensors_.end()) {
        tensors_[index].attr = slang::type::tensor_attr::kCONSTANT;
        if (length <= ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES) {
            tensors_[index].data = constant_copy_[index].data();
            tensors_[index].data_length = constant_copy_[index].size();
        } else {
            tensors_[index].data = buffer;
            tensors_[index].data_length = length;
        }
    } else {
        scalars_[index].data = constant_copy_[index];
    }

    return ANEURALNETWORKS_NO_ERROR;
}

int Model::SetOperandValueFromMemory(int32_t index, const Memory* memory, size_t offset,
                                     size_t length) {
    if (finished_) {
        std::cout << "Error: can not modify a finished model." << std::endl;
        return ANEURALNETWORKS_BAD_STATE;
    }
    if (index >= operand_id_) {
        std::cout << "ANeuralNetworksModel_setOperandValueFromMemory get an invalid index"
                  << std::endl;
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (length > 0xFFFFFFFF) {
        std::cout << "ANeuralNetworksModel_setOperandValueFromMemory value length of " << length
                  << " exceeds max size" << std::endl;
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (memory == nullptr) return ANEURALNETWORKS_BAD_DATA;

    if (tensors_.find(index) != tensors_.end()) {
        tensors_[index].attr = slang::type::tensor_attr::kCONSTANT;
        tensors_[index].data = (uint8_t*)memory->Data() + offset;
        tensors_[index].data_length = length;
    } else {
        std::cout << "ANeuralNetworksModel_setOperandValueFromMemory get an invalid index"
                  << std::endl;
        return ANEURALNETWORKS_BAD_DATA;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int Model::AddOperation(ANeuralNetworksOperationType type, uint32_t inputCount,
                        const uint32_t* inputs, uint32_t outputCount, const uint32_t* outputs) {
    if (finished_) {
        std::cout << "Error: can not modify a finished model." << std::endl;
        return ANEURALNETWORKS_BAD_STATE;
    }
    switch (type) {
        case ANEURALNETWORKS_ABS:
            op_creators_.push_back(std::make_shared<AbsCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_ADD:
            op_creators_.push_back(std::make_shared<AddCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_ARGMAX:
            op_creators_.push_back(std::make_shared<ArgmaxCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_ARGMIN:
            op_creators_.push_back(std::make_shared<ArgminCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_AVERAGE_POOL_2D:
            op_creators_.push_back(std::make_shared<AveragePool2DCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_BATCH_TO_SPACE_ND:
            op_creators_.push_back(std::make_shared<BatchToSpaceCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_CAST:
            op_creators_.push_back(std::make_shared<CastCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_CHANNEL_SHUFFLE:
            op_creators_.push_back(std::make_shared<ChannelShuffleCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_CONCATENATION:
            op_creators_.push_back(std::make_shared<ConcatenationCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_CONV_2D:
            op_creators_.push_back(std::make_shared<Conv2DCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_DEQUANTIZE:
            op_creators_.push_back(std::make_shared<DequantizeCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_DEPTHWISE_CONV_2D:
            op_creators_.push_back(std::make_shared<DepthwiseConv2DCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_DEPTH_TO_SPACE:
            op_creators_.push_back(std::make_shared<DepthToSpaceCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_DIV:
            op_creators_.push_back(std::make_shared<DivCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_EQUAL:
            op_creators_.push_back(std::make_shared<EqualCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_EXP:
            op_creators_.push_back(std::make_shared<ExpCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_EXPAND_DIMS:
            op_creators_.push_back(std::make_shared<ExpandDimsCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_ELU:
            op_creators_.push_back(std::make_shared<EluCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_FLOOR:
            op_creators_.push_back(std::make_shared<FloorCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_FULLY_CONNECTED:
            op_creators_.push_back(std::make_shared<FullyConnectedCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_GATHER:
            op_creators_.push_back(std::make_shared<GatherCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_GREATER:
            op_creators_.push_back(std::make_shared<GreaterCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_GREATER_EQUAL:
            op_creators_.push_back(std::make_shared<GreaterEqualCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_GROUPED_CONV_2D:
            op_creators_.push_back(std::make_shared<GroupedConv2DCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_HARD_SWISH:
            op_creators_.push_back(std::make_shared<HardSwishCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_INSTANCE_NORMALIZATION:
            op_creators_.push_back(std::make_shared<InstanceNormalizationCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_L2_NORMALIZATION:
            op_creators_.push_back(std::make_shared<L2NormalizationCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_LESS_EQUAL:
            op_creators_.push_back(std::make_shared<LessEqualCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_LESS:
            op_creators_.push_back(std::make_shared<LessCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_LOG:
            op_creators_.push_back(std::make_shared<LogCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION:
            op_creators_.push_back(std::make_shared<LocalResponseNormalizationCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_LOGISTIC:
            op_creators_.push_back(std::make_shared<LogisticCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_LOGICAL_NOT:
            op_creators_.push_back(std::make_shared<LogicalNotCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_LOGICAL_AND:
            op_creators_.push_back(std::make_shared<LogicalAndCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_LOGICAL_OR:
            op_creators_.push_back(std::make_shared<LogicalOrCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_L2_POOL_2D:
            op_creators_.push_back(std::make_shared<L2Pool2DCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_MAX_POOL_2D:
            op_creators_.push_back(std::make_shared<MaxPool2DCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_MEAN:
            op_creators_.push_back(std::make_shared<MeanCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_MUL:
            op_creators_.push_back(std::make_shared<MulCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_NEG:
            op_creators_.push_back(std::make_shared<NegCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_NOT_EQUAL:
            op_creators_.push_back(std::make_shared<NotEqualCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_PAD:
            op_creators_.push_back(std::make_shared<PadCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_PAD_V2:
            op_creators_.push_back(std::make_shared<PadV2Creator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_POW:
            op_creators_.push_back(std::make_shared<PowCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_PRELU:
            op_creators_.push_back(std::make_shared<PreluCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_QUANTIZE:
            op_creators_.push_back(std::make_shared<QuantizeCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_REDUCE_ALL:
            op_creators_.push_back(std::make_shared<ReduceAllCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_REDUCE_ANY:
            op_creators_.push_back(std::make_shared<ReduceAnyCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_REDUCE_MAX:
            op_creators_.push_back(std::make_shared<ReduceMaxCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_REDUCE_MIN:
            op_creators_.push_back(std::make_shared<ReduceMinCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_REDUCE_PROD:
            op_creators_.push_back(std::make_shared<ReduceProdCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_REDUCE_SUM:
            op_creators_.push_back(std::make_shared<ReduceSumCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_RELU:
            op_creators_.push_back(std::make_shared<ReluCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_RELU1:
            op_creators_.push_back(std::make_shared<Relu1Creator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_RELU6:
            op_creators_.push_back(std::make_shared<Relu6Creator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_RESHAPE:
            op_creators_.push_back(std::make_shared<ReshapeCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_RESIZE_BILINEAR:
            op_creators_.push_back(std::make_shared<ResizeBilinearCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_RSQRT:
            op_creators_.push_back(std::make_shared<RsqrtCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_SELECT:
            op_creators_.push_back(std::make_shared<SelectCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_SIN:
            op_creators_.push_back(std::make_shared<SinCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_SLICE:
            op_creators_.push_back(std::make_shared<SliceCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_SOFTMAX:
            op_creators_.push_back(std::make_shared<SoftmaxCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_SPACE_TO_DEPTH:
            op_creators_.push_back(std::make_shared<SpaceToDepthCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_SPACE_TO_BATCH_ND:
            op_creators_.push_back(std::make_shared<SpaceToBatchCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_SQUEEZE:
            op_creators_.push_back(std::make_shared<SqueezeCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_SQRT:
            op_creators_.push_back(std::make_shared<SqrtCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_STRIDED_SLICE:
            op_creators_.push_back(std::make_shared<StridedSliceCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_SUB:
            op_creators_.push_back(std::make_shared<SubCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_TANH:
            op_creators_.push_back(std::make_shared<TanhCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_TRANSPOSE:
            op_creators_.push_back(std::make_shared<TransposeCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        case ANEURALNETWORKS_TRANSPOSE_CONV_2D:
            op_creators_.push_back(std::make_shared<TransposeConv2DCreator>(
                    std::vector<uint32_t>(inputs, inputs + inputCount),
                    std::vector<uint32_t>(outputs, outputs + outputCount), tensors_, scalars_));
            break;
        default:
            std::cout << "operation is not supported" << std::endl;
            return ANEURALNETWORKS_BAD_DATA;
    }
    auto op = op_creators_.back();
    if (op->support_state_ == false) {
        return ANEURALNETWORKS_BAD_DATA;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int Model::IdentifyInputsAndOutputs(uint32_t inputCount, const uint32_t* inputs,
                                    uint32_t outputCount, const uint32_t* outputs) {
    if (finished_) {
        std::cout << "Error: can not modify a finished model." << std::endl;
        return ANEURALNETWORKS_BAD_STATE;
    }
    inputs_ = std::vector<uint32_t>(inputs, inputs + inputCount);
    outputs_ = std::vector<uint32_t>(outputs, outputs + outputCount);
    return ANEURALNETWORKS_NO_ERROR;
}

int Model::GetSupportedOperations(bool* supported_ops) const {
    for (int i = 0; i < op_creators_.size(); ++i) {
        supported_ops[i] = op_creators_[i]->Check();
        std::cout << "op " << op_creators_[i]->Type() << " support status: " << supported_ops[i]
                  << std::endl;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

}  // namespace sl
}  // namespace android
}  // namespace vsi