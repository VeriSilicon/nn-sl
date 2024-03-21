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

#include "MapOperation.h"

#include <algorithm>

#include "Types.h"
#include "Utils.h"
#include "tim/vx/ops.h"

namespace vsi::android::sl {

static VxTensor fuseActivation(VxGraph graph, int32_t fuseCode, VxTensor output) {
    VxOp op;
    switch (static_cast<slang::type::activation_type>(fuseCode)) {
        case slang::type::activation_type::kNONE:
            return output;
        case slang::type::activation_type::kRELU:
            op = graph->CreateOperation<tim::vx::ops::Relu>();
            break;
        case slang::type::activation_type::kRELU1:
            op = graph->CreateOperation<tim::vx::ops::Relu1>();
            break;
        case slang::type::activation_type::kRELU6:
            op = graph->CreateOperation<tim::vx::ops::Relu6>();
            break;
        default:
            LOGE("Unkown fuse code");
            return nullptr;
    }
    auto input = graph->CreateTensor(output->GetSpec().AsTransientSpec());
    op->BindInput(input);
    op->BindOutput(output);
    return input;
}

static Shape alignShape(const Shape& origShape, const Shape& refShape) {
    Shape expandedShape(origShape);
    size_t refRank = refShape.size();
    size_t origRank = origShape.size();
    for (size_t i = 0; i < refRank; ++i) {
        if (i >= origRank) {
            expandedShape.push_back(1);
        }
    }
    return expandedShape;
}

static tim::vx::Quantization computeBiasQuant(tim::vx::Quantization inputQuant,
                                              tim::vx::Quantization weightQuant) {
    auto weightQuantType = weightQuant.Type();
    if (weightQuantType == tim::vx::QuantType::NONE) {
        return {};
    }

    float inputScale = inputQuant.Scales().front();
    auto weightScales = weightQuant.Scales();
    std::vector<float> biasScales(weightScales.size());
    std::vector<int32_t> biasZeroPoints(weightScales.size(), 0);

    for (size_t i = 0; i < weightScales.size(); i++) {
        biasScales[i] = inputScale * weightScales[i];
    }

    return {weightQuantType, 0, biasScales, biasZeroPoints};
}

int mapOneInputOneOutput(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                         const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                         const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                         const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];

    auto input = vxTensors.at(idx_in);
    auto output = vxTensors.at(idx_out);

    auto op = opCreator->Lowering(graph);
    op->BindInput(input);
    op->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapBatchMatmul(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                   const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                   const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                   const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_in2 = inputs[1];
    uint32_t idx_out = outputs[0];

    auto input = vxTensors.at(idx_in);
    auto input2 = vxTensors.at(idx_in2);
    auto output = vxTensors.at(idx_out);

    auto op = opCreator->Lowering(graph);
    op->BindInput(input);
    op->BindInput(input2);
    op->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapConcatenation(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                     const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                     const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                     const std::vector<uint32_t>& outputs) {
    uint32_t idx_out = outputs[0];
    std::vector<VxTensor> timvx_inputs;
    for (int i = 0; i < inputs.size() - 1; ++i) {
        uint32_t idx_in = inputs[i];

        timvx_inputs.push_back(vxTensors.at(idx_in));
    }

    auto output = vxTensors.at(idx_out);

    auto concatenation = opCreator->Lowering(graph);
    concatenation->BindInputs(timvx_inputs);
    concatenation->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapConv2D(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
              const TensorMap& tensorMap, const ScalarMap& scalarMap,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_kernel = inputs[1];
    uint32_t idx_bias = inputs[2];
    uint32_t idx_out = outputs[0];
    uint32_t idx_act;
    inputs.size() == 7 || scalarMap.at(inputs.at(7)).dtype == slang::type::data_type::kBOOL8
            ? idx_act = inputs[6]
            : idx_act = inputs[9];

    auto input = vxTensors.at(idx_in);
    auto kernel = vxTensors.at(idx_kernel);
    auto bias = vxTensors.at(idx_bias);
    auto output = vxTensors.at(idx_out);

    auto inputQuant = input->GetSpec().GetQuantization();
    auto weightQuant = kernel->GetSpec().GetQuantization();
    auto biasQuant = computeBiasQuant(inputQuant, weightQuant);
    bias->GetSpec().SetQuantization(biasQuant);

    auto activationCodeScalar = scalarMap.at(idx_act);
    int32_t fuse_code = *reinterpret_cast<const int32_t*>(activationCodeScalar.data.data());

    output = fuseActivation(graph, fuse_code, output);

    auto conv2d = opCreator->Lowering(graph);
    conv2d->BindInput(input);
    conv2d->BindInput(kernel);
    conv2d->BindInput(bias);
    conv2d->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapDepthwiseConv2D(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                       const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                       const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                       const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_kernel = inputs[1];
    uint32_t idx_bias = inputs[2];
    uint32_t idx_out = outputs[0];
    uint32_t idx_act;
    inputs.size() == 8 || scalarMap.at(inputs.at(8)).dtype == slang::type::data_type::kBOOL8
            ? idx_act = inputs[7]
            : idx_act = inputs[10];

    auto input = vxTensors.at(idx_in);
    auto kernel = vxTensors.at(idx_kernel);
    auto bias = vxTensors.at(idx_bias);
    auto output = vxTensors.at(idx_out);

    auto inputQuant = input->GetSpec().GetQuantization();
    auto weightQuant = kernel->GetSpec().GetQuantization();
    auto biasQuant = computeBiasQuant(inputQuant, weightQuant);
    bias->GetSpec().SetQuantization(biasQuant);

    const uint8_t* p_act_code = scalarMap.at(idx_act).data.data();
    int32_t fuse_code = *(int32_t*)p_act_code;

    output = fuseActivation(graph, fuse_code, output);

    auto conv2d = opCreator->Lowering(graph);
    conv2d->BindInput(input);
    conv2d->BindInput(kernel);
    conv2d->BindInput(bias);
    conv2d->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapEltwise(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
               const TensorMap& tensorMap, const ScalarMap& scalarMap,
               const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_in2 = inputs[2];
    uint32_t idx_out = outputs[0];

    auto input = vxTensors.at(idx_in);
    auto input1 = vxTensors.at(idx_in1);
    auto output = vxTensors.at(idx_out);

    const uint8_t* fuse_code_data = scalarMap.at(idx_in2).data.data();
    int32_t fuse_code = *reinterpret_cast<const int32_t*>(fuse_code_data);

    output = fuseActivation(graph, fuse_code, output);
    if (output == nullptr) return ANEURALNETWORKS_BAD_DATA;

    auto in_shape = input->GetShape();
    auto in_shape1 = input1->GetShape();
    auto out_shape = output->GetShape();
    if (in_shape < out_shape && input->GetSpec().GetTensorAttribute() != tim::vx::CONSTANT)
        input->GetSpec().SetShape(alignShape(in_shape, out_shape));
    if (in_shape1 < out_shape && input1->GetSpec().GetTensorAttribute() != tim::vx::CONSTANT)
        input1->GetSpec().SetShape(alignShape(in_shape1, out_shape));

    auto eltwise = opCreator->Lowering(graph);
    eltwise->BindInput(input);
    eltwise->BindInput(input1);
    eltwise->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapEltwiseWithNoAct(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                        const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                        const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                        const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_out = outputs[0];

    auto input = vxTensors.at(idx_in);
    auto input1 = vxTensors.at(idx_in1);
    auto output = vxTensors.at(idx_out);

    auto eltwise_with_no_act = opCreator->Lowering(graph);
    eltwise_with_no_act->BindInput(input);
    eltwise_with_no_act->BindInput(input1);
    eltwise_with_no_act->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapEmbeddingLookup(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                       const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                       const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                       const std::vector<uint32_t>& outputs) {
    uint32_t idx_lookups = inputs[0];
    uint32_t idx_values = inputs[1];
    uint32_t idx_out = outputs[0];

    auto lookups = vxTensors.at(idx_lookups);
    auto values = vxTensors.at(idx_values);
    auto output = vxTensors.at(idx_out);

    auto embedding_lookup = opCreator->Lowering(graph);
    embedding_lookup->BindInput(lookups);
    embedding_lookup->BindInput(values);
    embedding_lookup->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapFullyConnected(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                      const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                      const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                      const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_weight = inputs[1];
    uint32_t idx_bias = inputs[2];
    uint32_t idx_act = inputs[3];
    uint32_t idx_out = outputs[0];

    auto input = vxTensors.at(idx_in);
    auto weight = vxTensors.at(idx_weight);
    auto bias = vxTensors.at(idx_bias);
    auto output = vxTensors.at(idx_out);

    if (input->GetShape().size() > 2 ||
        (input->GetShape().size() == 2 && input->GetShape()[0] != weight->GetShape()[0])) {
        uint32_t input_size = weight->GetShape()[0];
        uint32_t total_input_size = 1;
        for (int i = 0; i < input->GetShape().size(); i++) {
            total_input_size *= input->GetShape()[i];
        }
        uint32_t input_batch = total_input_size / input_size;
        auto reshape_output = graph->CreateTensor(input->GetSpec().AsTransientSpec());
        std::vector<uint32_t> new_shape{input_size, input_batch};
        auto reshape = graph->CreateOperation<tim::vx::ops::Reshape>(new_shape);
        (*reshape).BindInput(input);
        (*reshape).BindOutput(reshape_output);
        input = reshape_output;
    }

    const int32_t* p_act_code = (int32_t*)scalarMap.at(idx_act).data.data();
    output = fuseActivation(graph, *p_act_code, output);

    auto fc = opCreator->Lowering(graph);
    fc->BindInput(input);
    fc->BindInput(weight);
    fc->BindInput(bias);
    fc->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapGather(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
              const TensorMap& tensorMap, const ScalarMap& scalarMap,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_indices = inputs[2];
    uint32_t idx_out = outputs[0];

    auto input = vxTensors.at(idx_in);
    auto indices = vxTensors.at(idx_indices);
    auto output = vxTensors.at(idx_out);

    auto gather = opCreator->Lowering(graph);
    gather->BindInput(input);
    gather->BindInput(indices);
    gather->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapGroupedConv2d(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                     const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                     const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                     const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_kernel = inputs[1];
    uint32_t idx_bias = inputs[2];
    uint32_t idx_out = outputs[0];
    uint32_t idx_act;
    inputs.size() == 9 ? idx_act = inputs[7] : idx_act = inputs[10];

    auto input = vxTensors.at(idx_in);
    auto kernel = vxTensors.at(idx_kernel);
    auto bias = vxTensors.at(idx_bias);
    auto output = vxTensors.at(idx_out);

    auto inputQuant = input->GetSpec().GetQuantization();
    auto weightQuant = kernel->GetSpec().GetQuantization();
    auto biasQuant = computeBiasQuant(inputQuant, weightQuant);
    bias->GetSpec().SetQuantization(biasQuant);

    const uint8_t* p_act_code = scalarMap.at(idx_act).data.data();
    int32_t fuse_code = *(int32_t*)p_act_code;

    output = fuseActivation(graph, fuse_code, output);

    auto grouped_conv2d = opCreator->Lowering(graph);
    grouped_conv2d->BindInput(input);
    grouped_conv2d->BindInput(kernel);
    grouped_conv2d->BindInput(bias);
    grouped_conv2d->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapInstanceNormalization(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                             const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                             const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                             const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_gamma = inputs[1];
    uint32_t idx_beta = inputs[2];
    uint32_t idx_out = outputs[0];
    auto gammaScalar = scalarMap.at(idx_gamma);
    auto betaScalar = scalarMap.at(idx_beta);

    float gammaVal;
    float betaVal;
    if (gammaScalar.dtype == slang::type::data_type::kFP16) {
        const __fp16* gammaData = reinterpret_cast<const __fp16*>(gammaScalar.data.data());
        const __fp16* betaData = reinterpret_cast<const __fp16*>(betaScalar.data.data());
        gammaVal = static_cast<float>(*gammaData);
        betaVal = static_cast<float>(*betaData);
    } else {
        const float* gammaData = reinterpret_cast<const float*>(gammaScalar.data.data());
        const float* betaData = reinterpret_cast<const float*>(betaScalar.data.data());
        gammaVal = *gammaData;
        betaVal = *betaData;
    }

    auto input = vxTensors.at(idx_in);
    auto output = vxTensors.at(idx_out);
    VxTensor gamma;
    VxTensor beta;

    auto inputShape = input->GetShape();
    auto instanceNormOpCreator = std::dynamic_pointer_cast<InstanceNormalizationCreator>(opCreator);
    auto instance_normalization = instanceNormOpCreator->Lowering(graph);
    auto layout = instanceNormOpCreator->getLayout();
    uint32_t numChannels = 0;

    if (layout == tim::vx::DataLayout::WHCN) {
        numChannels = inputShape[2];
    } else if (layout == tim::vx::DataLayout::CWHN) {
        numChannels = inputShape[0];
    }

    if (input->GetDataType() == tim::vx::DataType::FLOAT16) {
        __fp16 gammaValFp16 = static_cast<__fp16>(gammaVal);
        __fp16 betaValFp16 = static_cast<__fp16>(betaVal);

        std::vector<__fp16> gammaValBroadcasted(numChannels, gammaValFp16);
        std::vector<__fp16> betaValBroadcasted(numChannels, betaValFp16);

        tim::vx::TensorSpec spec(tim::vx::DataType::FLOAT16, {numChannels},
                                 tim::vx::TensorAttribute::CONSTANT);
        gamma = graph->CreateTensor(spec, gammaValBroadcasted.data());
        beta = graph->CreateTensor(spec, betaValBroadcasted.data());
    } else {
        std::vector<float> gammaValBroadcasted(numChannels, gammaVal);
        std::vector<float> betaValBroadcasted(numChannels, betaVal);

        tim::vx::TensorSpec spec(tim::vx::DataType::FLOAT32, {numChannels},
                                 tim::vx::TensorAttribute::CONSTANT);
        gamma = graph->CreateTensor(spec, gammaValBroadcasted.data());
        beta = graph->CreateTensor(spec, betaValBroadcasted.data());
    }

    instance_normalization->BindInput(input);
    instance_normalization->BindInput(beta);
    instance_normalization->BindInput(gamma);
    instance_normalization->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapHashtableLookup(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                       const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                       const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                       const std::vector<uint32_t>& outputs) {
    uint32_t idx_lookups = inputs[0];
    uint32_t idx_keys = inputs[1];
    uint32_t idx_values = inputs[2];
    uint32_t idx_out = outputs[0];
    uint32_t idx_hits = outputs[1];

    auto lookups = vxTensors.at(idx_lookups);
    auto keys = vxTensors.at(idx_keys);
    auto values = vxTensors.at(idx_values);
    auto output = vxTensors.at(idx_out);
    auto hits = vxTensors.at(idx_hits);
    auto hashtable_lookup = opCreator->Lowering(graph);
    hashtable_lookup->BindInput(lookups);
    hashtable_lookup->BindInput(keys);
    hashtable_lookup->BindInput(values);
    hashtable_lookup->BindOutput(output);
    hashtable_lookup->BindOutput(hits);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapLogicalAndOr(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                    const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                    const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                    const std::vector<uint32_t>& outputs) {
    uint32_t idx_in0 = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_out = outputs[0];

    auto input0 = vxTensors.at(idx_in0);
    auto input1 = vxTensors.at(idx_in1);
    auto output = vxTensors.at(idx_out);

    auto logical_and_or = opCreator->Lowering(graph);
    logical_and_or->BindInput(input0);
    logical_and_or->BindInput(input1);
    logical_and_or->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapPack(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
            const TensorMap& tensorMap, const ScalarMap& scalarMap,
            const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    int32_t inputs_num = inputs.size();
    uint32_t idx_out = outputs[0];
    std::vector<VxTensor> inputs_tensors;
    for (int i = 1; i < inputs_num; ++i) {
        uint32_t idx_in = inputs[i];

        inputs_tensors.push_back(vxTensors.at(idx_in));
    }

    auto output = vxTensors.at(idx_out);

    auto pack = opCreator->Lowering(graph);
    pack->BindInputs(inputs_tensors);
    pack->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapPool2D(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
              const TensorMap& tensorMap, const ScalarMap& scalarMap,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    uint32_t idx_act;
    (inputs.size() == 7 || inputs.size() == 8) ? idx_act = inputs[6] : idx_act = inputs[9];

    auto input = vxTensors.at(idx_in);
    auto output = vxTensors.at(idx_out);
    const uint8_t* p_act_code = scalarMap.at(idx_act).data.data();
    int32_t fuse_code = *(int32_t*)p_act_code;

    output = fuseActivation(graph, fuse_code, output);

    auto pool2d = opCreator->Lowering(graph);
    pool2d->BindInput(input);
    pool2d->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapPrelu(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
             const TensorMap& tensorMap, const ScalarMap& scalarMap,
             const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_alpha = inputs[1];
    uint32_t idx_out = outputs[0];

    auto input = vxTensors.at(idx_in);
    auto alpha = vxTensors.at(idx_alpha);
    auto alpha_shape = alpha->GetShape();
    bool dims_all_1 = std::all_of(alpha_shape.begin(), alpha_shape.end(),
                                  [](uint32_t dims) { return dims == 1; });
    if (dims_all_1) alpha->GetSpec().SetShape({1});
    auto output = vxTensors.at(idx_out);

    auto prelu = opCreator->Lowering(graph);
    prelu->BindInput(input);
    prelu->BindInput(alpha);
    prelu->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapRelationalOp(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                    const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                    const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                    const std::vector<uint32_t>& outputs) {
    uint32_t idx_in0 = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_out = outputs[0];

    auto input0 = vxTensors.at(idx_in0);
    auto input1 = vxTensors.at(idx_in1);
    auto output = vxTensors.at(idx_out);

    auto relational_op = opCreator->Lowering(graph);
    relational_op->BindInput(input0);
    relational_op->BindInput(input1);
    relational_op->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapRoi(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
           const TensorMap& tensorMap, const ScalarMap& scalarMap,
           const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_regions = inputs[1];
    uint32_t idx_batch_index = inputs[2];
    uint32_t idx_out = outputs[0];

    auto input = vxTensors.at(idx_in);
    auto regions = vxTensors.at(idx_regions);
    auto batch_index = vxTensors.at(idx_batch_index);
    auto output = vxTensors.at(idx_out);

    auto roi_align = opCreator->Lowering(graph);
    roi_align->BindInput(input);
    roi_align->BindInput(regions);
    roi_align->BindInput(batch_index);
    roi_align->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapSelect(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
              const TensorMap& tensorMap, const ScalarMap& scalarMap,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_choose = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_in2 = inputs[2];
    uint32_t idx_out = outputs[0];

    auto choose = vxTensors.at(idx_choose);
    auto input1 = vxTensors.at(idx_in1);
    auto input2 = vxTensors.at(idx_in2);
    auto output = vxTensors.at(idx_out);

    auto select = opCreator->Lowering(graph);
    select->BindInput(choose);
    select->BindInput(input1);
    select->BindInput(input2);
    select->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapSplit(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
             const TensorMap& tensorMap, const ScalarMap& scalarMap,
             const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_num_splits = inputs[2];
    std::vector<VxTensor> outputs_tensors;
    auto p_num_splits = scalarMap.at(idx_num_splits).data.data();
    int num_splits = *(int32_t*)p_num_splits;

    for (int i = 0; i < num_splits; ++i) {
        uint32_t idx_out = outputs[i];
        outputs_tensors.push_back(vxTensors.at(idx_out));
    }

    auto input = vxTensors.at(idx_in);

    auto split = opCreator->Lowering(graph);
    split->BindInput(input);
    split->BindOutputs(outputs_tensors);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapSvdf(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
            const TensorMap& tensorMap, const ScalarMap& scalarMap,
            const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_weights_feature = inputs[1];
    uint32_t idx_weights_time = inputs[2];
    uint32_t idx_state_out = outputs[0];
    uint32_t idx_out = outputs[1];
    uint32_t idx_bias, idx_state_in, idx_act;
    int32_t fuse_code = 0;
    VxTensor bias;
    if (tensorMap.at(inputs[3]).shape.size() == 1) {
        idx_bias = inputs[3];
        idx_state_in = inputs[4];
        bias = vxTensors.at(idx_bias);
        if (inputs.size() == 7) {
            idx_act = inputs.back();
            auto p_act = scalarMap.at(idx_act).data.data();
            fuse_code = *(int32_t*)p_act;
        }
    } else {
        idx_state_in = inputs[3];
        if (inputs.size() == 6) {
            idx_act = inputs.back();
            auto p_act = scalarMap.at(idx_act).data.data();
            fuse_code = *(int32_t*)p_act;
        }
    }

    auto input = vxTensors.at(idx_in);
    auto weights_feature = vxTensors.at(idx_weights_feature);
    auto weights_time = vxTensors.at(idx_weights_time);
    auto state_in = vxTensors.at(idx_state_in);
    auto state_out = vxTensors.at(idx_state_out);
    auto output = vxTensors.at(idx_out);

    auto svdf = opCreator->Lowering(graph);
    svdf->BindInput(input);
    svdf->BindInput(state_in);
    svdf->BindInput(weights_feature);
    svdf->BindInput(weights_time);
    if (tensorMap.at(inputs[3]).shape.size() == 1) {
        svdf->BindInput(bias);
    }
    svdf->BindOutput(output);
    svdf->BindOutput(state_out);
    return ANEURALNETWORKS_NO_ERROR;
}

int mapTopK(VxGraph graph, std::shared_ptr<OpCreator> opCreator, const VxTensorMap& vxTensors,
            const TensorMap& tensorMap, const ScalarMap& scalarMap,
            const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    uint32_t idx_indices = outputs[1];

    auto input = vxTensors.at(idx_in);
    auto output = vxTensors.at(idx_out);
    auto indices = vxTensors.at(idx_indices);

    auto top_k = opCreator->Lowering(graph);
    top_k->BindInput(input);
    top_k->BindOutput(output);
    top_k->BindOutput(indices);

    return ANEURALNETWORKS_NO_ERROR;
}

int mapTransposeConv2d(VxGraph graph, std::shared_ptr<OpCreator> opCreator,
                       const VxTensorMap& vxTensors, const TensorMap& tensorMap,
                       const ScalarMap& scalarMap, const std::vector<uint32_t>& inputs,
                       const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_kernel = inputs[1];
    uint32_t idx_bias = inputs[2];
    uint32_t idx_out = outputs[0];
    uint32_t idx_act;
    inputs.size() == 9 ? idx_act = inputs[7] : idx_act = inputs[9];

    auto input = vxTensors.at(idx_in);
    auto kernel = vxTensors.at(idx_kernel);
    auto bias = vxTensors.at(idx_bias);
    auto output = vxTensors.at(idx_out);

    auto inputQuant = input->GetSpec().GetQuantization();
    auto weightQuant = kernel->GetSpec().GetQuantization();
    auto biasQuant = computeBiasQuant(inputQuant, weightQuant);
    bias->GetSpec().SetQuantization(biasQuant);

    const uint8_t* p_act_code = scalarMap.at(idx_act).data.data();
    int32_t fuse_code = *(int32_t*)p_act_code;

    output = fuseActivation(graph, fuse_code, output);

    auto transpose_conv2d = opCreator->Lowering(graph);
    transpose_conv2d->BindInput(input);
    transpose_conv2d->BindInput(kernel);
    transpose_conv2d->BindInput(bias);
    transpose_conv2d->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

}  // namespace vsi::android::sl