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

#include "Model.h"

#include "Utils.h"

namespace vsi::android::sl {

int Model::addOperand(const ANeuralNetworksOperandType& type) {
    if (finished_) {
        LOGE("Model::addOperand cannot modify a finished model");
        return ANEURALNETWORKS_BAD_STATE;
    }

    auto operandType = static_cast<OperandType>(type.type);
    if (operandType == OperandType::TENSOR_FLOAT32 || operandType == OperandType::TENSOR_FLOAT16 ||
        operandType == OperandType::TENSOR_INT32 || operandType == OperandType::TENSOR_BOOL8 ||
        operandType == OperandType::TENSOR_QUANT8_ASYMM ||
        operandType == OperandType::TENSOR_QUANT8_SYMM ||
        operandType == OperandType::TENSOR_QUANT8_ASYMM_SIGNED ||
        operandType == OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL ||
        operandType == OperandType::TENSOR_QUANT16_ASYMM ||
        operandType == OperandType::TENSOR_QUANT16_SYMM) {
        // Implies tensor.
        if (type.dimensionCount == 0) {
            LOGE("Model::addOperand passed a tensor but has zero rank");
            return ANEURALNETWORKS_BAD_DATA;
        }
        auto shape = std::vector<uint32_t>(type.dimensions, type.dimensions + type.dimensionCount);
        slang::type::tensor_storage tensor = {
                .dtype = MapDataType(type.type),
                .qtype = MapQuantType(type.type),
                .shape = shape,
                .scale = type.scale,
                .zero_point = type.zeroPoint,
        };
        tensors_.insert({numOperands_, tensor});
    } else if (operandType == OperandType::FLOAT32 || operandType == OperandType::FLOAT16 ||
               operandType == OperandType::INT32 || operandType == OperandType::UINT32 ||
               operandType == OperandType::BOOL) {
        // Implies scalar.
        if (type.dimensionCount != 0) {
            LOGE("Model::addOperand passed a scalar but has non-zero rank");
            return ANEURALNETWORKS_BAD_DATA;
        }
        slang::type::scalar_storage scalar = {.dtype = MapDataType(type.type)};
        scalars_.insert({numOperands_, scalar});
    } else {
        LOGW("Model::addOperand passed an operand with unsupported type: %d", operandType);
    }

    numOperands_++;
    return ANEURALNETWORKS_NO_ERROR;
}

int Model::setOperandSymmPerChannelQuantParams(
        int32_t index, const ANeuralNetworksSymmPerChannelQuantParams& channelQuant) {
    if (finished_) {
        LOGE("Model::setOperandSymmPerChannelQuantParams cannot modify a finished model");
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (index < 0 || index >= numOperands_) {
        LOGE("Model::setOperandSymmPerChannelQuantParams passed an invalid operand index");
        return ANEURALNETWORKS_BAD_DATA;
    }

    if (tensors_.find(index) != tensors_.end()) {
        // reverse channel_dim axis
        uint32_t channelDim = tensors_[index].shape.size() - channelQuant.channelDim - 1;
        tensors_[index].channel_dim = channelDim;
        tensors_[index].per_channel_scales.assign(channelQuant.scales,
                                                  channelQuant.scales + channelQuant.scaleCount);
        tensors_[index].per_channel_zero_points.assign(channelQuant.scaleCount, 0);
    } else {
        LOGE("Model::setOperandSymmPerChannelQuantParams passed an invalid operand index");
        return ANEURALNETWORKS_BAD_DATA;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int Model::setOperandValue(int32_t index, const void* buffer, size_t length) {
    if (finished_) {
        LOGE("Model::setOperandValue cannot modify a finished model");
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (index < 0 || index >= numOperands_) {
        LOGE("Model::setOperandValue passed an invalid operand index");
        return ANEURALNETWORKS_BAD_DATA;
    }

    if (buffer == nullptr) {
        LOGW("Model::setOperandValue operand (%d) is marked as optional", index);
        return ANEURALNETWORKS_NO_ERROR;
    }

    if (length <= ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES) {
        size_t storageOffset = constantCopyStorage_.size();
        size_t alignedLength = alignSize(length, 4);

        constantCopyStorage_.resize(storageOffset + alignedLength);

        uint8_t* storageBuffer = constantCopyStorage_.data() + storageOffset;
        std::copy_n(reinterpret_cast<const uint8_t*>(buffer), length, storageBuffer);

        operandValueInfos_[index] = {
                .size = length,
                .offset = storageOffset,
                .buffer = nullptr,
        };

        if (auto it = tensors_.find(index); it != tensors_.end()) {
            auto& [_, tensor] = *it;
            tensor.data.assign(storageBuffer, storageBuffer + length);
        }

        if (auto it = scalars_.find(index); it != scalars_.end()) {
            auto& [_, scalar] = *it;
            scalar.data.assign(storageBuffer, storageBuffer + length);
        }
    } else {
        operandValueInfos_[index] = {
                .size = length,
                .buffer = buffer,
        };
    }

    if (auto it = tensors_.find(index); it != tensors_.end()) {
        auto& [_, tensor] = *it;
        tensor.attr = slang::type::tensor_attr::kCONSTANT;
    }

    return ANEURALNETWORKS_NO_ERROR;
}

int Model::setOperandValueFromMemory(int32_t index, const IMemory* memory, size_t offset,
                                     size_t length) {
    if (finished_) {
        LOGE("Model::setOperandValueFromMemory cannot modify a finished model");
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (index < 0 || index >= numOperands_) {
        LOGE("Model::setOperandValueFromMemory passed an invalid operand index");
        return ANEURALNETWORKS_BAD_DATA;
    }

    int status = memory->validate(nullptr, IOType::NONE, index, nullptr, offset, length);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        LOGE("Model::setOperandValueFromMemory failed to validate memory");
        return status;
    }

    if (length <= ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES) {
        auto mapping = memory->map();
        if (mapping.getStatus() != ANEURALNETWORKS_NO_ERROR) {
            LOGE("Model::setOperandValueFromMemory failed to map memory");
            return mapping.getStatus();
        }

        const uint8_t* data = reinterpret_cast<uint8_t*>(mapping.getData()) + offset;

        if (auto it = tensors_.find(index); it != tensors_.end()) {
            auto& [_, tensor] = *it;
            tensor.data.assign(data, data + length);
        }

        if (auto it = scalars_.find(index); it != scalars_.end()) {
            auto& [_, scalar] = *it;
            scalar.data.assign(data, data + length);
        }
    }

    operandValueInfos_[index] = {
            .offset = offset,
            .memory = memory,
    };

    if (auto it = tensors_.find(index); it != tensors_.end()) {
        auto& [_, tensor] = *it;
        tensor.attr = slang::type::tensor_attr::kCONSTANT;
    }

    return ANEURALNETWORKS_NO_ERROR;
}

int Model::setOperandValueFromModel(int32_t index, const Model* reference) {
    if (finished_) {
        LOGE("Model::setOperandValueFromModel cannot modify a finished model");
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (!reference->isFinished()) {
        LOGE("Model::setOperandValueFromModel reference model is not finished");
        return ANEURALNETWORKS_BAD_STATE;
    }
    referenceModels_.push_back(reference);
    return ANEURALNETWORKS_NO_ERROR;
}

int Model::addOperation(ANeuralNetworksOperationType type, uint32_t inputCount,
                        const uint32_t* inputs, uint32_t outputCount, const uint32_t* outputs) {
    if (finished_) {
        LOGE("Model::addOperation cannot modify a finished model");
        return ANEURALNETWORKS_BAD_STATE;
    }

    auto opInputs = std::vector<uint32_t>(inputs, inputs + inputCount);
    auto opOutputs = std::vector<uint32_t>(outputs, outputs + outputCount);

    bool hasEmptyScalar = std::any_of(opInputs.cbegin(), opInputs.cend(), [this](uint32_t i) {
        if (auto it = scalars_.find(i); it != scalars_.cend()) {
            auto [_, scalar] = *it;
            return scalar.data.empty();
        }
        return false;
    });

    if (hasEmptyScalar) {
        LOGW("Model::addOperation OP type: %d has empty input scalars", type);
        opCreators_.push_back(std::make_shared<PlaceHolderOpCreator>(type));
        opSupported_.push_back(false);
        return ANEURALNETWORKS_NO_ERROR;
    }

    std::shared_ptr<OpCreator> opCreator;
    switch (type) {
        case ANEURALNETWORKS_ABS:
            opCreator = std::make_shared<AbsCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_ADD:
            opCreator = std::make_shared<AddCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_ARGMAX:
            opCreator = std::make_shared<ArgmaxCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_ARGMIN:
            opCreator = std::make_shared<ArgminCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_AVERAGE_POOL_2D:
            opCreator =
                    std::make_shared<AveragePool2DCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_BATCH_MATMUL:
            opCreator =
                    std::make_shared<BatchMatmulCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_BATCH_TO_SPACE_ND:
            opCreator =
                    std::make_shared<BatchToSpaceCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_CAST:
            opCreator = std::make_shared<CastCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_CHANNEL_SHUFFLE:
            opCreator = std::make_shared<ChannelShuffleCreator>(opInputs, opOutputs, tensors_,
                                                                scalars_);
            break;
        case ANEURALNETWORKS_CONCATENATION:
            opCreator =
                    std::make_shared<ConcatenationCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_CONV_2D:
            opCreator = std::make_shared<Conv2DCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_DEQUANTIZE:
            opCreator =
                    std::make_shared<DequantizeCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_DEPTHWISE_CONV_2D:
            opCreator = std::make_shared<DepthwiseConv2DCreator>(opInputs, opOutputs, tensors_,
                                                                 scalars_);
            break;
        case ANEURALNETWORKS_DEPTH_TO_SPACE:
            opCreator =
                    std::make_shared<DepthToSpaceCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_DIV:
            opCreator = std::make_shared<DivCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_EMBEDDING_LOOKUP:
            opCreator = std::make_shared<EmbeddingLookupCreator>(opInputs, opOutputs, tensors_,
                                                                 scalars_);
            break;
        case ANEURALNETWORKS_EQUAL:
            opCreator = std::make_shared<EqualCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_EXP:
            opCreator = std::make_shared<ExpCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_EXPAND_DIMS:
            opCreator =
                    std::make_shared<ExpandDimsCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_ELU:
            opCreator = std::make_shared<EluCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_FLOOR:
            opCreator = std::make_shared<FloorCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_FULLY_CONNECTED:
            opCreator = std::make_shared<FullyConnectedCreator>(opInputs, opOutputs, tensors_,
                                                                scalars_);
            break;
        case ANEURALNETWORKS_GATHER:
            opCreator = std::make_shared<GatherCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_GREATER:
            opCreator = std::make_shared<GreaterCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_GREATER_EQUAL:
            opCreator =
                    std::make_shared<GreaterEqualCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_GROUPED_CONV_2D:
            opCreator =
                    std::make_shared<GroupedConv2DCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_HASHTABLE_LOOKUP:
            opCreator = std::make_shared<HashtableLookupCreator>(opInputs, opOutputs, tensors_,
                                                                 scalars_);
            break;
        case ANEURALNETWORKS_HARD_SWISH:
            opCreator = std::make_shared<HardSwishCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_INSTANCE_NORMALIZATION:
            opCreator = std::make_shared<InstanceNormalizationCreator>(opInputs, opOutputs,
                                                                       tensors_, scalars_);
            break;
        case ANEURALNETWORKS_L2_NORMALIZATION:
            opCreator = std::make_shared<L2NormalizationCreator>(opInputs, opOutputs, tensors_,
                                                                 scalars_);
            break;
        case ANEURALNETWORKS_LESS_EQUAL:
            opCreator = std::make_shared<LessEqualCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_LESS:
            opCreator = std::make_shared<LessCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_LOG:
            opCreator = std::make_shared<LogCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION:
            opCreator = std::make_shared<LocalResponseNormalizationCreator>(opInputs, opOutputs,
                                                                            tensors_, scalars_);
            break;
        case ANEURALNETWORKS_LOGISTIC:
            opCreator = std::make_shared<LogisticCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_LOGICAL_NOT:
            opCreator =
                    std::make_shared<LogicalNotCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_LOGICAL_AND:
            opCreator =
                    std::make_shared<LogicalAndCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_LOGICAL_OR:
            opCreator = std::make_shared<LogicalOrCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_LOG_SOFTMAX:
            opCreator =
                    std::make_shared<LogSoftmaxCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_L2_POOL_2D:
            opCreator = std::make_shared<L2Pool2DCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_MAX_POOL_2D:
            opCreator = std::make_shared<MaxPool2DCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_MEAN:
            opCreator = std::make_shared<MeanCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_MAXIMUM:
            opCreator = std::make_shared<MaximumCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_MINIMUM:
            opCreator = std::make_shared<MinimumCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_MIRROR_PAD:
            opCreator = std::make_shared<MirrorPadCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_MUL:
            opCreator = std::make_shared<MulCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_NEG:
            opCreator = std::make_shared<NegCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_NOT_EQUAL:
            opCreator = std::make_shared<NotEqualCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_PACK:
            opCreator = std::make_shared<PackCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_PAD:
            opCreator = std::make_shared<PadCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_PAD_V2:
            opCreator = std::make_shared<PadV2Creator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_POW:
            opCreator = std::make_shared<PowCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_PRELU:
            opCreator = std::make_shared<PreluCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_QUANTIZE:
            opCreator = std::make_shared<QuantizeCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_REDUCE_ALL:
            opCreator = std::make_shared<ReduceAllCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_REDUCE_ANY:
            opCreator = std::make_shared<ReduceAnyCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_REDUCE_MAX:
            opCreator = std::make_shared<ReduceMaxCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_REDUCE_MIN:
            opCreator = std::make_shared<ReduceMinCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_REDUCE_PROD:
            opCreator =
                    std::make_shared<ReduceProdCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_REDUCE_SUM:
            opCreator = std::make_shared<ReduceSumCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_RELU:
            opCreator = std::make_shared<ReluCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_RELU1:
            opCreator = std::make_shared<Relu1Creator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_RELU6:
            opCreator = std::make_shared<Relu6Creator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_RESHAPE:
            opCreator = std::make_shared<ReshapeCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_RESIZE_BILINEAR:
            opCreator = std::make_shared<ResizeBilinearCreator>(opInputs, opOutputs, tensors_,
                                                                scalars_);
            break;
        case ANEURALNETWORKS_REVERSE:
            opCreator = std::make_shared<ReverseCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR:
            opCreator =
                    std::make_shared<ResizeNearestCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_ROI_ALIGN:
            opCreator = std::make_shared<RoiAlignCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        // case ANEURALNETWORKS_ROI_POOLING:  // roi_pooling not support at present
        //     op_creators_.push_back(std::make_shared<RoiPoolingCreator>(
        //             inputsList,
        //             outputsList, tensors_, scalars_);
        //     break;
        case ANEURALNETWORKS_RSQRT:
            opCreator = std::make_shared<RsqrtCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_SELECT:
            opCreator = std::make_shared<SelectCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_SIN:
            opCreator = std::make_shared<SinCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_SLICE:
            opCreator = std::make_shared<SliceCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_SOFTMAX:
            opCreator = std::make_shared<SoftmaxCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_SPACE_TO_DEPTH:
            opCreator =
                    std::make_shared<SpaceToDepthCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_SPACE_TO_BATCH_ND:
            opCreator =
                    std::make_shared<SpaceToBatchCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_SPLIT:
            opCreator = std::make_shared<SplitCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_SQUEEZE:
            opCreator = std::make_shared<SqueezeCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_SQRT:
            opCreator = std::make_shared<SqrtCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_STRIDED_SLICE:
            opCreator =
                    std::make_shared<StridedSliceCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_SUB:
            opCreator = std::make_shared<SubCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        // case ANEURALNETWORKS_SVDF:  // svdf not support at present
        //     opCreator = std::make_shared<SvdfCreator>(opInputs, opOutputs, tensors_, scalars_);
        //     break;
        case ANEURALNETWORKS_TANH:
            opCreator = std::make_shared<TanhCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_TILE:
            opCreator = std::make_shared<TileCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_TOPK_V2:
            opCreator = std::make_shared<TopKCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_TRANSPOSE:
            opCreator = std::make_shared<TransposeCreator>(opInputs, opOutputs, tensors_, scalars_);
            break;
        case ANEURALNETWORKS_TRANSPOSE_CONV_2D:
            opCreator = std::make_shared<TransposeConv2DCreator>(opInputs, opOutputs, tensors_,
                                                                 scalars_);
            break;
        default:
            opCreator = std::make_shared<PlaceHolderOpCreator>(type);
            break;
    }

    opCreators_.push_back(opCreator);
    opSupported_.push_back(opCreator->isSupported());
    return ANEURALNETWORKS_NO_ERROR;
}

int Model::relaxComputationFloat32toFloat16(bool relaxed) {
    if (finished_) {
        LOGE("Model::relaxComputationFloat32toFloat16 cannot modify a finished model");
        return ANEURALNETWORKS_BAD_STATE;
    }
    relaxed_ = relaxed;
    return ANEURALNETWORKS_NO_ERROR;
}

int Model::finish() {
    if (finished_) {
        LOGE("Model::finish the model is already finished");
        return ANEURALNETWORKS_BAD_STATE;
    }
    finished_ = true;
    return ANEURALNETWORKS_NO_ERROR;
}

int Model::identifyInputsAndOutputs(uint32_t inputCount, const uint32_t* inputs,
                                    uint32_t outputCount, const uint32_t* outputs) {
    if (finished_) {
        LOGE("Model::identifyInputsAndOutputs cannot modify a finished model");
        return ANEURALNETWORKS_BAD_STATE;
    }
    inputs_ = std::vector<uint32_t>(inputs, inputs + inputCount);
    outputs_ = std::vector<uint32_t>(outputs, outputs + outputCount);
    return ANEURALNETWORKS_NO_ERROR;
}

int Model::getSupportedOperations(bool* supportedOps) const {
    if (!finished_) {
        LOGE("Model::getSupportedOperations the model is unfinished");
        return ANEURALNETWORKS_BAD_STATE;
    }

    LOGV("Model::getSupportedOperations SL graph total ops count: %zu", opCreators_.size());
    for (size_t i = 0; i < opCreators_.size(); i++) {
        supportedOps[i] = opCreators_[i]->checkSupported() && opSupported_[i];
        LOGV("Model::getSupportedOperations op index: %zu, type: %d, supported: %d", i,
             opCreators_[i]->getType(), supportedOps[i]);
    }
    return ANEURALNETWORKS_NO_ERROR;
}

}  // namespace vsi::android::sl