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

#include "Execution.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <thread>

#include "MapOperation.h"
#include "Memory.h"
#include "Utils.h"
#include "tim/transform/layout_inference.h"
#include "tim/vx/platform/platform.h"
#include "tim/vx/tensor.h"

namespace vsi::android::sl {

Execution::Execution(Compilation* compilation)
    : compilation_(compilation), reusable_(false), measure_(false), state_(State::PREPARATION) {
    if (auto graph = compilation->getCompiledGraph(); graph != nullptr) {
        inputVxTensors_ = graph->InputsTensor();
        outputVxTensors_ = graph->OutputsTensor();
        runtimeGraph_ = std::move(graph);
    }

    relaxed_ = compilation->getModel()->isRelaxed();
    timeoutDuration_ = Duration::min();
    loopTimeoutDuration_ = Duration::min();
}

int Execution::setReusable(bool reusable) {
    if (state_ != State::PREPARATION) {
        LOGE("Execution::setReusable the execution may only be modified in the preparation state");
        return ANEURALNETWORKS_BAD_STATE;
    }

    reusable_ = reusable;
    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::setTimeout(Duration duration) {
    if (state_ != State::PREPARATION) {
        LOGE("Execution::setTimeout the execution may only be modified in the preparation state");
        return ANEURALNETWORKS_BAD_STATE;
    }

    timeoutDuration_ = duration;
    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::setLoopTimeout(Duration duration) {
    if (state_ != State::PREPARATION) {
        LOGE("Execution::setLoopTimeout the execution may only be modified in the preparation "
             "state");
        return ANEURALNETWORKS_BAD_STATE;
    }

    loopTimeoutDuration_ = duration;
    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::setMeasureTiming(bool measure) {
    if (state_ != State::PREPARATION) {
        LOGE("Execution::setMeasureTiming the execution may only be modified in the preparation "
             "state");
        return ANEURALNETWORKS_BAD_STATE;
    }

    measure_ = measure;
    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::setInput(int32_t index, const ANeuralNetworksOperandType* type, const void* buffer,
                        size_t length) {
    if (state_ != State::PREPARATION) {
        LOGE("Execution::setInput the execution may only be modified in the preparation state");
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (type != nullptr) {
        auto* model = compilation_->getModel();
        uint32_t input = model->getInputs()[index];
        auto& tensorMap = model->getTensorMap();
        auto& inputTensor = tensorMap[input];
        if (inputTensor.dtype != MapDataType(type->type) ||
            std::fabs(inputTensor.scale - type->scale) > std::numeric_limits<float>::epsilon() ||
            inputTensor.zero_point != type->zeroPoint) {
            LOGE("Execution::setInput get invalid ANeuralNetworksOperandType");
            return ANEURALNETWORKS_BAD_DATA;
        }

        auto shape =
                std::vector<uint32_t>(type->dimensions, type->dimensions + type->dimensionCount);
        inputTensor.shape = shape;
    }

    IOBufferInfo inputBufferInfo = {
            .offset = 0,
            .length = length,
            .buffer = const_cast<void*>(buffer),
    };
    inputBufferInfos_.push_back(inputBufferInfo);

    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::setOutput(int32_t index, const ANeuralNetworksOperandType* type, void* buffer,
                         size_t length) {
    if (state_ != State::PREPARATION) {
        LOGE("Execution::setOutput the execution may only be modified in the preparation state");
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (type != nullptr) {
        auto* model = compilation_->getModel();
        uint32_t output = model->getOutputs()[index];
        auto& tensorMap = model->getTensorMap();
        auto& outputTensor = tensorMap[output];
        if (outputTensor.dtype != MapDataType(type->type) ||
            std::fabs(outputTensor.scale - type->scale) > std::numeric_limits<float>::epsilon() ||
            outputTensor.zero_point != type->zeroPoint) {
            LOGE("Execution::setOutput get invalid ANeuralNetworksOperandType");
            return ANEURALNETWORKS_BAD_DATA;
        }

        auto shape =
                std::vector<uint32_t>(type->dimensions, type->dimensions + type->dimensionCount);
        outputTensor.shape = shape;
    }

    IOBufferInfo outputBufferInfo = {
            .offset = 0,
            .length = length,
            .buffer = buffer,
    };
    outputBufferInfos_.push_back(outputBufferInfo);

    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::setInputFromMemory(int32_t index, const ANeuralNetworksOperandType* type,
                                  const IMemory* memory, size_t offset, size_t length) {
    if (state_ != State::PREPARATION) {
        LOGE("Execution::setInputFromMemory the execution may only be modified in the preparation "
             "state");
        return ANEURALNETWORKS_BAD_STATE;
    }

    int status = memory->validate(compilation_, IOType::INPUT, index, type, offset, length);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        LOGE("Execution::setInputFromMemory failed to validate memory");
        return status;
    }

    if (type != nullptr && type->dimensionCount != 0) {  // implies tensor
        auto* model = compilation_->getModel();
        uint32_t input = model->getInputs()[index];
        auto& tensorMap = model->getTensorMap();
        auto& inputTensor = tensorMap[input];

        auto shape =
                std::vector<uint32_t>(type->dimensions, type->dimensions + type->dimensionCount);
        inputTensor.shape = shape;
    }

    IOBufferInfo inputBufferInfo = {
            .offset = offset,
            .length = length,
            .memory = memory,
    };
    inputBufferInfos_.push_back(inputBufferInfo);

    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::setOutputFromMemory(int32_t index, const ANeuralNetworksOperandType* type,
                                   const IMemory* memory, size_t offset, size_t length) {
    if (state_ != State::PREPARATION) {
        LOGE("Execution::setInputFromMemory the execution may only be modified in the preparation "
             "state");
        return ANEURALNETWORKS_BAD_STATE;
    }

    int status = memory->validate(compilation_, IOType::OUTPUT, index, type, offset, length);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        LOGE("Execution::setInputFromMemory failed to validate memory");
        return status;
    }

    if (type != nullptr) {
        auto* model = compilation_->getModel();
        uint32_t output = model->getOutputs()[index];
        auto& tensorMap = model->getTensorMap();
        auto& outputTensor = tensorMap[output];

        auto shape =
                std::vector<uint32_t>(type->dimensions, type->dimensions + type->dimensionCount);
        outputTensor.shape = shape;
    }

    IOBufferInfo outputBufferInfo = {
            .offset = offset,
            .length = length,
            .memory = memory,
    };
    outputBufferInfos_.push_back(outputBufferInfo);

    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::compute() {
    if (state_ == State::COMPLETED && !reusable_) {
        LOGE("Execution::compute try to schedule multiple computations for an execution which is "
             "not reusable");
        return ANEURALNETWORKS_BAD_STATE;
    }
    state_ = State::COMPUTATION;

    // This function will be called multiple times, need to judge whether it is the first call.
    if (runtimeGraph_ == nullptr) {
        int result = compile();
        if (result != ANEURALNETWORKS_NO_ERROR) {
            LOGE("Execution::compute failed to compile graph for the 1st time");

            state_ = State::COMPLETED;
            return result;
        }
    }

    if (inputVxTensors_.size() != inputBufferInfos_.size()) {
        LOGE("Execution::compute not all inputs have set buffer or memory");
        state_ = State::COMPLETED;
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (outputVxTensors_.size() != outputBufferInfos_.size()) {
        LOGE("Execution::compute not all outputs have set buffer or memory");
        state_ = State::COMPLETED;
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (!compilation_->isBurst()) {
        for (size_t i = 0; i < inputVxTensors_.size(); i++) {
            auto inputVxTensor = inputVxTensors_[i];
            auto inputBufferInfo = inputBufferInfos_[i];

            if (const auto* memory = inputBufferInfo.memory; memory != nullptr) {
                if (!memory->isInitialized()) {
                    LOGE("Execution::compute input memory is uninitialized");
                    return ANEURALNETWORKS_OP_FAILED;
                }

                void* data = reinterpret_cast<uint8_t*>(memory->getData()) + inputBufferInfo.offset;
                size_t length =
                        (inputBufferInfo.length == 0) ? memory->getSize() : inputBufferInfo.length;

                if (!inputVxTensor->CopyDataToTensor(data, length)) {
                    LOGE("Execution::compute failed to copy input data from memory");

                    state_ = State::COMPLETED;
                    return ANEURALNETWORKS_BAD_STATE;
                }
            } else if (const void* buffer = inputBufferInfo.buffer; buffer != nullptr) {
                if (!inputVxTensor->CopyDataToTensor(buffer, inputBufferInfo.length)) {
                    LOGE("Execution::compute failed to copy input data from user buffer");

                    state_ = State::COMPLETED;
                    return ANEURALNETWORKS_BAD_STATE;
                }
            } else {
                LOGW("Execution::compute input:%zu has null buffer or memory", i);
                continue;
            }
        }
    }

    if (!runtimeGraph_->Run()) {
        LOGE("Execution::compute failed to run tim-vx graph");

        state_ = State::COMPLETED;
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (!compilation_->isBurst()) {
        for (size_t i = 0; i < outputVxTensors_.size(); i++) {
            auto outputVxTensor = outputVxTensors_[i];
            auto outputBufferInfo = outputBufferInfos_[i];

            if (const auto* memory = outputBufferInfo.memory; memory != nullptr) {
                void* data =
                        reinterpret_cast<uint8_t*>(memory->getData()) + outputBufferInfo.offset;
                if (!outputVxTensor->CopyDataFromTensor(data)) {
                    LOGE("Execution::compute failed to copy output data to memory");

                    state_ = State::COMPLETED;
                    return ANEURALNETWORKS_BAD_STATE;
                }

                const_cast<IMemory*>(memory)->setInitialized(true);
            } else if (void* buffer = outputBufferInfo.buffer; buffer != nullptr) {
                if (!outputVxTensor->CopyDataFromTensor(buffer)) {
                    LOGE("Execution::compute failed to copy output data to user buffer");

                    state_ = State::COMPLETED;
                    return ANEURALNETWORKS_BAD_STATE;
                }
            } else {
                LOGE("Execution::compute output:%zu has null buffer or memory", i);
                state_ = State::COMPLETED;
                return ANEURALNETWORKS_BAD_STATE;
            }
        }
    }

    state_ = State::COMPLETED;
    return ANEURALNETWORKS_NO_ERROR;
}

CallbackEvent* Execution::createSyncEvent() {
    auto deadline = timeoutDuration_ != Duration::min() ? Clock::now() + timeoutDuration_
                                                        : TimePoint::max();
    auto* event = new CallbackEvent(deadline);
    syncEvent_ = event;
    return event;
}

int Execution::startCompute() {
    if (state_ == State::COMPLETED && !reusable_) {
        LOGE("Execution::startCompute try to schedule multiple computations for an execution which "
             "is "
             "not reusable");

        state_ = State::COMPLETED;
        return ANEURALNETWORKS_BAD_STATE;
    }
    state_ = State::COMPUTATION;

    // This function will be called multiple times, need to judge whether it is the first call.
    if (runtimeGraph_ == nullptr) {
        int result = compile();
        if (result != ANEURALNETWORKS_NO_ERROR) {
            LOGE("Execution::startCompute failed to compile graph for the 1st time");

            state_ = State::COMPLETED;
            return result;
        }
    }

    if (inputVxTensors_.size() != inputBufferInfos_.size()) {
        LOGE("Execution::startCompute not all inputs have set buffer or memory");
        state_ = State::COMPLETED;
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (outputVxTensors_.size() != outputBufferInfos_.size()) {
        LOGE("Execution::startCompute not all outputs have set buffer or memory");
        state_ = State::COMPLETED;
        return ANEURALNETWORKS_BAD_STATE;
    }

    auto asyncThread = std::thread([this]() {
        for (size_t i = 0; i < inputVxTensors_.size(); i++) {
            auto inputVxTensor = inputVxTensors_[i];
            auto inputBufferInfo = inputBufferInfos_[i];

            if (const auto* memory = inputBufferInfo.memory; memory != nullptr) {
                if (!memory->isInitialized()) {
                    LOGE("Execution::startCompute input memory is uninitialized");
                    return ANEURALNETWORKS_OP_FAILED;
                }

                void* data = reinterpret_cast<uint8_t*>(memory->getData()) + inputBufferInfo.offset;
                size_t length =
                        (inputBufferInfo.length == 0) ? memory->getSize() : inputBufferInfo.length;
                if (!inputVxTensor->CopyDataToTensor(data, length)) {
                    LOGE("Execution::startCompute failed to copy input data from memory");

                    state_ = State::COMPLETED;
                    return ANEURALNETWORKS_OP_FAILED;
                }
            } else if (const void* buffer = inputBufferInfo.buffer; buffer != nullptr) {
                if (!inputVxTensor->CopyDataToTensor(buffer, inputBufferInfo.length)) {
                    LOGE("Execution::startCompute failed to copy input data from user buffer");

                    state_ = State::COMPLETED;
                    return ANEURALNETWORKS_OP_FAILED;
                }
            } else {
                LOGW("Execution::startCompute input:%zu has null buffer or memory", i);
                continue;
            }
        }

        if (!runtimeGraph_->Run()) {
            LOGE("Execution::startCompute failed to run tim-vx graph");

            state_ = State::COMPLETED;
            return ANEURALNETWORKS_OP_FAILED;
        }

        for (size_t i = 0; i < outputVxTensors_.size(); i++) {
            auto outputVxTensor = outputVxTensors_[i];
            auto outputBufferInfo = outputBufferInfos_[i];

            if (const auto* memory = outputBufferInfo.memory; memory != nullptr) {
                void* data =
                        reinterpret_cast<uint8_t*>(memory->getData()) + outputBufferInfo.offset;
                if (!outputVxTensor->CopyDataFromTensor(data)) {
                    LOGE("Execution::startCompute failed to copy output data to memory");

                    state_ = State::COMPLETED;
                    return ANEURALNETWORKS_OP_FAILED;
                }

                const_cast<IMemory*>(memory)->setInitialized(true);
            } else if (void* buffer = outputBufferInfo.buffer; buffer != nullptr) {
                if (!outputVxTensor->CopyDataFromTensor(buffer)) {
                    LOGE("Execution::startCompute failed to copy output data to user buffer");

                    state_ = State::COMPLETED;
                    return ANEURALNETWORKS_OP_FAILED;
                }
            } else {
                LOGE("Execution::startCompute output:%zu has null buffer or memory", i);
                state_ = State::COMPLETED;
                return ANEURALNETWORKS_OP_FAILED;
            }
        }

        syncEvent_->notify();
        state_ = State::COMPLETED;
        return ANEURALNETWORKS_NO_ERROR;
    });

    return syncEvent_->bindThread(std::move(asyncThread));
}

int Execution::getDuration(DurationCode durationCode, uint64_t* duration) const {
    if (state_ != State::COMPLETED) {
        LOGE("Execution::getDuration called when the execution is not in the completed state");
        return ANEURALNETWORKS_BAD_STATE;
    }

    switch (durationCode) {
        case ANEURALNETWORKS_DURATION_ON_HARDWARE:
        case ANEURALNETWORKS_DURATION_IN_DRIVER:
        case ANEURALNETWORKS_FENCED_DURATION_ON_HARDWARE:
        case ANEURALNETWORKS_FENCED_DURATION_IN_DRIVER:
            break;
        default:
            LOGE("Execution::getDuration passed an invalid duration code");
            return ANEURALNETWORKS_BAD_DATA;
    }

    // The driver does not support timing measurement for now.
    *duration = std::numeric_limits<uint64_t>::max();

    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::getOutputOperandRank(int32_t index, uint32_t* rank) const {
    if (state_ != State::COMPLETED) {
        LOGE("Execution::getOutputOperandRank called when the execution is not in the completed "
             "state");
        return ANEURALNETWORKS_BAD_STATE;
    }

    auto* model = compilation_->getModel();
    uint32_t output = model->getOutputs()[index];
    auto& tensorMap = model->getTensorMap();
    auto outputTensor = tensorMap[output];
    *rank = outputTensor.shape.size();
    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::getOutputOperandDimensions(int32_t index, uint32_t* dimensions) const {
    if (state_ != State::COMPLETED) {
        LOGE("Execution::getOutputOperandDimensions called when the execution is not in the "
             "completed state");
        return ANEURALNETWORKS_BAD_STATE;
    }

    auto* model = compilation_->getModel();
    uint32_t output = model->getOutputs()[index];
    auto& tensorMap = model->getTensorMap();
    auto outputTensor = tensorMap[output];

    const auto& shape = outputTensor.shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        dimensions[i] = outputTensor.shape[i];
    }
    return ANEURALNETWORKS_NO_ERROR;
}

VxTensor Execution::createVxConstantTensor(const slang::type::tensor_storage& tensor,
                                           Model::OperandValueInfo valueInfo) {
    tim::vx::DataType dtype = ToTvxDataType(tensor.dtype);
    tim::vx::ShapeType shape = tensor.shape;
    std::reverse(shape.begin(), shape.end());
    tim::vx::Quantization quantization;
    tim::vx::QuantType qtype = ToTvxQuantType(tensor.qtype);
    if (qtype == tim::vx::QuantType::ASYMMETRIC) {
        quantization = tim::vx::Quantization(qtype, tensor.scale, tensor.zero_point);
    } else if (qtype == tim::vx::QuantType::SYMMETRIC_PER_CHANNEL) {
        quantization = tim::vx::Quantization(qtype, tensor.channel_dim, tensor.per_channel_scales,
                                             tensor.per_channel_zero_points);
    }
    tim::vx::TensorSpec spec(dtype, shape, tim::vx::TensorAttribute::CONSTANT, quantization);

    const void* data = nullptr;
    const auto* memory = valueInfo.memory;
    if (memory != nullptr) {
        int status = const_cast<IMemory*>(memory)->map();
        if (status != ANEURALNETWORKS_NO_ERROR) {
            LOGE("Execution::createVxConstantTensor failed to map memory");
            return nullptr;
        }
        data = reinterpret_cast<uint8_t*>(memory->getData()) + valueInfo.offset;
    } else {
        data = valueInfo.buffer;
    }

    VxTensor vxTensor = nullptr;
    if (relaxed_ && spec.GetDataType() == tim::vx::DataType::FLOAT32) {
        auto shape = spec.shape_;
        size_t numElements = std::reduce(shape.cbegin(), shape.cend(), 1, std::multiplies<>());

        // NOLINTBEGIN(modernize-*)
        const float* fp32Data = reinterpret_cast<const float*>(data);
        std::unique_ptr<__fp16[]> fp16Buffer = std::make_unique<__fp16[]>(numElements);
        // NOLINTEND(modernize-*)

        std::transform(fp32Data, fp32Data + numElements, fp16Buffer.get(),
                       [](float val) { return static_cast<__fp16>(val); });

        spec.SetDataType(tim::vx::DataType::FLOAT16);
        vxTensor = vxGraph_->CreateTensor(spec, fp16Buffer.get());
    } else {
        vxTensor = vxGraph_->CreateTensor(spec, data);
    }

    if (memory != nullptr) {
        const_cast<IMemory*>(memory)->unmap();
    }

    return vxTensor;
}

VxTensor Execution::createVxIOTensor(const slang::type::tensor_storage& tensor,
                                     tim::vx::TensorAttribute attr) {
    tim::vx::DataType dtype = ToTvxDataType(tensor.dtype);
    tim::vx::ShapeType shape = tensor.shape;
    std::reverse(shape.begin(), shape.end());
    tim::vx::Quantization quantization;
    tim::vx::QuantType qtype = ToTvxQuantType(tensor.qtype);
    if (qtype == tim::vx::QuantType::ASYMMETRIC) {
        quantization = tim::vx::Quantization(qtype, tensor.scale, tensor.zero_point);
    } else if (qtype == tim::vx::QuantType::SYMMETRIC_PER_CHANNEL) {
        quantization = tim::vx::Quantization(qtype, tensor.channel_dim, tensor.per_channel_scales,
                                             tensor.per_channel_zero_points);
    }
    tim::vx::TensorSpec spec(dtype, shape, attr, quantization);
    return vxGraph_->CreateIOTensor(spec);
}

VxTensor Execution::createTransientTensor(const slang::type::tensor_storage& tensor) {
    tim::vx::DataType dtype = ToTvxDataType(tensor.dtype);
    tim::vx::ShapeType shape = tensor.shape;
    std::reverse(shape.begin(), shape.end());
    tim::vx::Quantization quantization;
    tim::vx::QuantType qtype = ToTvxQuantType(tensor.qtype);
    if (qtype == tim::vx::QuantType::ASYMMETRIC) {
        quantization = tim::vx::Quantization(qtype, tensor.scale, tensor.zero_point);
    } else if (qtype == tim::vx::QuantType::SYMMETRIC_PER_CHANNEL) {
        quantization = tim::vx::Quantization(qtype, tensor.channel_dim, tensor.per_channel_scales,
                                             tensor.per_channel_zero_points);
    }
    tim::vx::TensorSpec spec(dtype, shape, tim::vx::TensorAttribute::TRANSIENT, quantization);

    if (relaxed_ && spec.GetDataType() == tim::vx::DataType::FLOAT32) {
        spec.SetDataType(tim::vx::DataType::FLOAT16);
    }
    return vxGraph_->CreateTensor(spec);
}

int Execution::mapOperations(const std::vector<std::shared_ptr<OpCreator>>& opCreators,
                             const TensorMap& tensorMap, const ScalarMap& scalarMap) {
    for (const auto& opCreator : opCreators) {
        const auto& inputs = opCreator->getInputs();
        const auto& outputs = opCreator->getOutputs();
        int result = ANEURALNETWORKS_NO_ERROR;
        switch (opCreator->getType()) {
            case ANEURALNETWORKS_ABS:
            case ANEURALNETWORKS_ARGMAX:
            case ANEURALNETWORKS_ARGMIN:
            case ANEURALNETWORKS_BATCH_TO_SPACE_ND:
            case ANEURALNETWORKS_CAST:
            case ANEURALNETWORKS_CHANNEL_SHUFFLE:
            case ANEURALNETWORKS_DEPTH_TO_SPACE:
            case ANEURALNETWORKS_DEQUANTIZE:
            case ANEURALNETWORKS_ELU:
            case ANEURALNETWORKS_EXP:
            case ANEURALNETWORKS_EXPAND_DIMS:
            case ANEURALNETWORKS_FLOOR:
            case ANEURALNETWORKS_HARD_SWISH:
            case ANEURALNETWORKS_L2_NORMALIZATION:
            case ANEURALNETWORKS_LOG:
            case ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION:
            case ANEURALNETWORKS_LOGISTIC:
            case ANEURALNETWORKS_LOGICAL_NOT:
            case ANEURALNETWORKS_LOG_SOFTMAX:
            case ANEURALNETWORKS_MEAN:
            case ANEURALNETWORKS_MIRROR_PAD:
            case ANEURALNETWORKS_NEG:
            case ANEURALNETWORKS_PAD:
            case ANEURALNETWORKS_PAD_V2:
            case ANEURALNETWORKS_QUANTIZE:
            case ANEURALNETWORKS_REDUCE_ALL:
            case ANEURALNETWORKS_REDUCE_ANY:
            case ANEURALNETWORKS_REDUCE_MAX:
            case ANEURALNETWORKS_REDUCE_MIN:
            case ANEURALNETWORKS_REDUCE_PROD:
            case ANEURALNETWORKS_REDUCE_SUM:
            case ANEURALNETWORKS_RELU:
            case ANEURALNETWORKS_RELU1:
            case ANEURALNETWORKS_RELU6:
            case ANEURALNETWORKS_RESHAPE:
            case ANEURALNETWORKS_RESIZE_BILINEAR:
            case ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR:
            case ANEURALNETWORKS_REVERSE:
            case ANEURALNETWORKS_RSQRT:
            case ANEURALNETWORKS_SIN:
            case ANEURALNETWORKS_SLICE:
            case ANEURALNETWORKS_SOFTMAX:
            case ANEURALNETWORKS_SPACE_TO_DEPTH:
            case ANEURALNETWORKS_SPACE_TO_BATCH_ND:
            case ANEURALNETWORKS_SQUEEZE:
            case ANEURALNETWORKS_SQRT:
            case ANEURALNETWORKS_STRIDED_SLICE:
            case ANEURALNETWORKS_TANH:
            case ANEURALNETWORKS_TILE:
            case ANEURALNETWORKS_TRANSPOSE:
                result = mapOneInputOneOutput(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                              inputs, outputs);
                break;
            case ANEURALNETWORKS_ADD:
                result = mapEltwise(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                    outputs);
                break;
            case ANEURALNETWORKS_AVERAGE_POOL_2D:
                result = mapPool2D(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                   outputs);
                break;
            case ANEURALNETWORKS_BATCH_MATMUL:
                result = mapBatchMatmul(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                        inputs, outputs);
                break;
            case ANEURALNETWORKS_CONCATENATION:
                result = mapConcatenation(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                          inputs, outputs);
                break;
            case ANEURALNETWORKS_CONV_2D:
                result = mapConv2D(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                   outputs);
                break;
            case ANEURALNETWORKS_DEPTHWISE_CONV_2D:
                result = mapDepthwiseConv2D(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                            inputs, outputs);
                break;
            case ANEURALNETWORKS_DIV:
                result = mapEltwise(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                    outputs);
                break;
            case ANEURALNETWORKS_EMBEDDING_LOOKUP:
                result = mapEmbeddingLookup(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                            inputs, outputs);
                break;
            case ANEURALNETWORKS_EQUAL:
                result = mapRelationalOp(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                         inputs, outputs);
                break;
            case ANEURALNETWORKS_FULLY_CONNECTED:
                result = mapFullyConnected(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                           inputs, outputs);
                break;
            case ANEURALNETWORKS_GATHER:
                result = mapGather(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                   outputs);
                break;
            case ANEURALNETWORKS_GREATER:
                result = mapRelationalOp(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                         inputs, outputs);
                break;
            case ANEURALNETWORKS_GREATER_EQUAL:
                result = mapRelationalOp(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                         inputs, outputs);
                break;
            case ANEURALNETWORKS_GROUPED_CONV_2D:
                result = mapGroupedConv2d(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                          inputs, outputs);
                break;
            case ANEURALNETWORKS_HASHTABLE_LOOKUP:
                result = mapHashtableLookup(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                            inputs, outputs);
                break;
            case ANEURALNETWORKS_INSTANCE_NORMALIZATION:
                result = mapInstanceNormalization(vxGraph_, opCreator, vxTensors_, tensorMap,
                                                  scalarMap, inputs, outputs);
                break;
            case ANEURALNETWORKS_LESS:
                result = mapRelationalOp(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                         inputs, outputs);
                break;
            case ANEURALNETWORKS_LESS_EQUAL:
                result = mapRelationalOp(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                         inputs, outputs);
                break;
            case ANEURALNETWORKS_LOGICAL_AND:
                result = mapLogicalAndOr(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                         inputs, outputs);
                break;
            case ANEURALNETWORKS_LOGICAL_OR:
                result = mapLogicalAndOr(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                         inputs, outputs);
                break;
            case ANEURALNETWORKS_L2_POOL_2D:
                result = mapPool2D(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                   outputs);
                break;
            case ANEURALNETWORKS_MAX_POOL_2D:
                result = mapPool2D(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                   outputs);
                break;
            case ANEURALNETWORKS_MAXIMUM:
                result = mapEltwiseWithNoAct(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                             inputs, outputs);
                break;
            case ANEURALNETWORKS_MINIMUM:
                result = mapEltwiseWithNoAct(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                             inputs, outputs);
                break;
            case ANEURALNETWORKS_MUL:
                result = mapEltwise(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                    outputs);
                break;
            case ANEURALNETWORKS_NOT_EQUAL:
                result = mapRelationalOp(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                         inputs, outputs);
                break;
            case ANEURALNETWORKS_PACK:
                result = mapPack(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                 outputs);
                break;
            case ANEURALNETWORKS_POW:
                result = mapEltwiseWithNoAct(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                             inputs, outputs);
                break;
            case ANEURALNETWORKS_PRELU:
                result = mapPrelu(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                  outputs);
                break;
            case ANEURALNETWORKS_ROI_ALIGN:
                result = mapRoi(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                outputs);
                break;
            // case ANEURALNETWORKS_ROI_POOLING:  // not support roi_pooling at present
            //     result = MapRoi(vx_graph_, op_creator, vx_tensors_, tensor_map, scalar_map,
            //     inputs,
            //                     outputs);
            //     break;
            case ANEURALNETWORKS_SELECT:
                result = mapSelect(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                   outputs);
                break;
            case ANEURALNETWORKS_SPLIT:
                result = mapSplit(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                  outputs);
                break;
            case ANEURALNETWORKS_SUB:
                result = mapEltwise(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                    outputs);
                break;
            case ANEURALNETWORKS_SVDF:
                result = mapSvdf(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                 outputs);
                break;
            case ANEURALNETWORKS_TOPK_V2:
                result = mapTopK(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap, inputs,
                                 outputs);
                break;
            case ANEURALNETWORKS_TRANSPOSE_CONV_2D:
                result = mapTransposeConv2d(vxGraph_, opCreator, vxTensors_, tensorMap, scalarMap,
                                            inputs, outputs);
                break;
            default:
                LOGE("Execution::mapOperation op type: %d not supported", opCreator->getType());
                result = ANEURALNETWORKS_BAD_STATE;
        }
        if (result != ANEURALNETWORKS_NO_ERROR) {
            return result;
        }
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int Execution::compile() {
    auto context = compilation_->getContext();
    const auto* model = compilation_->getModel();
    const auto& inputs = model->getInputs();
    const auto& outputs = model->getOutputs();
    const auto& tensorMap = model->getTensorMap();
    const auto& scalarMap = model->getScalarMap();
    const auto& operandValuesInfoMap = model->getOperandValueInfos();
    const auto& operations = model->getOpCreators();

    // Check for output tensor with dynamic axis.
    for (size_t i = 0; i < outputs.size(); i++) {
        uint32_t output = outputs[i];
        auto outputTensor = tensorMap.at(output);

        bool hasDynamicAxis = std::any_of(outputTensor.shape.begin(), outputTensor.shape.end(),
                                          [](uint32_t s) { return s == 0; });
        if (hasDynamicAxis) {
            LOGE("Execution::compile output:%zu has dynamic axis which is not supported", i);
            return ANEURALNETWORKS_OP_FAILED;
        }
    }

    vxGraph_ = context->CreateGraph();

    if (compilation_->getCacheState() == Compilation::CacheState::LOADED) {
        auto nbg = vxGraph_->CreateOperation<tim::vx::ops::NBG>(
                reinterpret_cast<const char*>(compilation_->getCacheData()), inputs.size(),
                outputs.size());
        for (uint32_t input : inputs) {
            auto inputVxTensor =
                    createVxIOTensor(tensorMap.at(input), tim::vx::TensorAttribute::INPUT);
            nbg->BindInput(inputVxTensor);
        }
        for (uint32_t output : outputs) {
            auto outputVxTensor =
                    createVxIOTensor(tensorMap.at(output), tim::vx::TensorAttribute::OUTPUT);
            nbg->BindOutput(outputVxTensor);
        }
    } else {
        // Create I/O vx tensors.
        for (uint32_t input : inputs) {
            auto inputVxTensor =
                    createVxIOTensor(tensorMap.at(input), tim::vx::TensorAttribute::INPUT);
            if (relaxed_ && inputVxTensor->GetDataType() == tim::vx::DataType::FLOAT32) {
                auto spec = inputVxTensor->GetSpec();
                spec.SetAttribute(tim::vx::TensorAttribute::TRANSIENT);
                spec.SetDataType(tim::vx::DataType::FLOAT16);
                auto inputVxTensorFp16Proxy = vxGraph_->CreateTensor(spec);

                auto dataConvert = vxGraph_->CreateOperation<tim::vx::ops::DataConvert>();
                dataConvert->BindInput(inputVxTensor);
                dataConvert->BindOutput(inputVxTensorFp16Proxy);

                vxTensors_[input] = inputVxTensorFp16Proxy;
            } else {
                vxTensors_[input] = inputVxTensor;
            }
        }
        for (uint32_t output : outputs) {
            auto outputVxTensor =
                    createVxIOTensor(tensorMap.at(output), tim::vx::TensorAttribute::OUTPUT);
            if (relaxed_ && outputVxTensor->GetDataType() == tim::vx::DataType::FLOAT32) {
                auto spec = outputVxTensor->GetSpec();
                spec.SetAttribute(tim::vx::TensorAttribute::TRANSIENT);
                spec.SetDataType(tim::vx::DataType::FLOAT16);
                auto outputVxTensorFp16Proxy = vxGraph_->CreateTensor(spec);

                auto dataConvert = vxGraph_->CreateOperation<tim::vx::ops::DataConvert>();
                dataConvert->BindInput(outputVxTensorFp16Proxy);
                dataConvert->BindOutput(outputVxTensor);

                vxTensors_[output] = outputVxTensorFp16Proxy;
            } else {
                vxTensors_[output] = outputVxTensor;
            }
        }

        // Create constant vx tensors.
        for (const auto& [operandIndex, tensor] : tensorMap) {
            if (auto it = operandValuesInfoMap.find(operandIndex);
                it != operandValuesInfoMap.end()) {
                auto [_, valueInfo] = *it;
                if (valueInfo.buffer == nullptr && valueInfo.memory == nullptr) {
                    valueInfo.buffer = model->getConstantCopyData(valueInfo.offset);
                }
                vxTensors_[operandIndex] = createVxConstantTensor(tensor, valueInfo);
            } else if (vxTensors_.count(operandIndex) == 0) {
                vxTensors_[operandIndex] = createTransientTensor(tensor);
            }
        }

        int result = mapOperations(operations, tensorMap, scalarMap);
        if (result != ANEURALNETWORKS_NO_ERROR) {
            LOGE("Execution::compile failed to map operations");
            return result;
        }

        auto [vxGraph, _] = tim::transform::LayoutInference(vxGraph_, context);
        auto inputTensors = vxGraph->InputsTensor();
        auto outputTensors = vxGraph->OutputsTensor();

        for (size_t i = 0; i < inputs.size(); i++) {
            uint32_t input = inputs[i];
            vxTensors_[input] = inputTensors[i];
        }
        for (size_t i = 0; i < outputs.size(); i++) {
            uint32_t output = outputs[i];
            vxTensors_[output] = outputTensors[i];
        }

        if (compilation_->getCacheState() == Compilation::CacheState::EMPTY) {
            size_t nbgSize;
            if (!vxGraph->CompileToBinary(nullptr, &nbgSize)) {
                LOGE("Execution::compile failed to compile tim-vx graph");
                return ANEURALNETWORKS_OP_FAILED;
            }

            std::vector<uint8_t> nbgBuffer(nbgSize);
            if (!vxGraph->CompileToBinary(nbgBuffer.data(), &nbgSize)) {
                LOGE("Execution::compile failed to compile tim-vx graph");
                return ANEURALNETWORKS_OP_FAILED;
            }

            compilation_->writeToCache(nbgBuffer.data(), nbgSize);
        }

        vxGraph_ = vxGraph;
    }

    if (!vxGraph_->Compile()) {
        LOGE("Execution::compile failed to compile tim-vx graph");
        return ANEURALNETWORKS_OP_FAILED;
    }

    compilation_->setCompiledGraph(vxGraph_);

    inputVxTensors_ = vxGraph_->InputsTensor();
    outputVxTensors_ = vxGraph_->OutputsTensor();
    runtimeGraph_ = vxGraph_;

    return ANEURALNETWORKS_NO_ERROR;
}

}  // namespace vsi::android::sl