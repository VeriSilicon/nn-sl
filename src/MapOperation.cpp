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

#include "Utils.h"
#include "tim/vx/ops.h"

namespace vsi {
namespace android {
namespace sl {
namespace {
std::shared_ptr<tim::vx::Tensor> CreateTvxTensor(std::shared_ptr<tim::vx::Graph> graph,
                                                 const slang::type::tensor_storage& tensor) {
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
    tim::vx::TensorSpec spec(data_type, shape, tim::vx::TensorAttribute::TRANSIENT, quantization);
    return graph->CreateTensor(spec);
}

std::shared_ptr<tim::vx::Tensor> FuseActivation(std::shared_ptr<tim::vx::Graph> graph,
                                                int32_t fuse_code,
                                                std::shared_ptr<tim::vx::Tensor> output) {
    std::shared_ptr<tim::vx::Operation> op;
    switch (static_cast<slang::type::activation_type>(fuse_code)) {
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

std::vector<uint32_t> ExpandedShape(
    const std::vector<uint32_t>& long_shape,
    const std::vector<uint32_t>& short_shape) {
  std::vector<uint32_t> expanded_shape(short_shape);
  int32_t ref_rank = long_shape.size();
  int32_t origin_rank = short_shape.size();
  for (int32_t i = 0; i < ref_rank; ++i) {
    if (i >= origin_rank) expanded_shape.push_back(1);
  }
  return expanded_shape;
}
}  // namespace

int MapOneInputOneOutput(std::shared_ptr<tim::vx::Graph> graph,
                         std::shared_ptr<OpCreator> op_creator,
                         std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                         const TensorMap& tensor_map, const ScalarMap& scalar_map,
                         const std::vector<uint32_t>& inputs,
                         const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];

    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }  // If not graph input/output, create const or transient tensor

    auto input = vx_tensors[idx_in];
    auto output = vx_tensors[idx_out];

    auto op = op_creator->Lowering(graph);
    op->BindInput(input);
    op->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapBatchMatmul(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                   std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                   const TensorMap& tensor_map, const ScalarMap& scalar_map,
                   const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_in2 = inputs[1];
    uint32_t idx_out = outputs[0];

    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_in2)) {
        vx_tensors.insert({idx_in2, CreateTvxTensor(graph, tensor_map.at(idx_in2))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }  // If not graph input/output, create const or transient tensor

    auto input = vx_tensors[idx_in];
    auto input2 = vx_tensors[idx_in2];
    auto output = vx_tensors[idx_out];

    auto op = op_creator->Lowering(graph);
    op->BindInput(input);
    op->BindInput(input2);
    op->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapConcatenation(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                     std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                     const TensorMap& tensor_map, const ScalarMap& scalar_map,
                     const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_out = outputs[0];
    std::vector<std::shared_ptr<tim::vx::Tensor>> timvx_inputs;
    for (int i = 0; i < inputs.size() - 1; ++i) {
        uint32_t idx_in = inputs[i];
        if (!vx_tensors.count(idx_in)) {
            vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
        }
        timvx_inputs.push_back(vx_tensors[idx_in]);
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }
    auto output = vx_tensors[idx_out];

    auto concatenation = op_creator->Lowering(graph);
    concatenation->BindInputs(timvx_inputs);
    concatenation->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapConv2D(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
              std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
              const TensorMap& tensor_map, const ScalarMap& scalar_map,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_kernel = inputs[1];
    uint32_t idx_bias = inputs[2];
    uint32_t idx_out = outputs[0];
    uint32_t idx_act;
    inputs.size() == 7 || scalar_map.at(inputs.at(7)).dtype == slang::type::data_type::kBOOL8
            ? idx_act = inputs[6]
            : idx_act = inputs[9];
    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    auto input_scale = vx_tensors[idx_in]->GetSpec().quantization_.Scales();

    if (!vx_tensors.count(idx_kernel)) {
        vx_tensors.insert({idx_kernel, CreateTvxTensor(graph, tensor_map.at(idx_kernel))});
    }
    tim::vx::QuantType quant_weight_type = vx_tensors[idx_kernel]->GetSpec().quantization_.Type();
    std::vector<float> weight_scales = vx_tensors[idx_kernel]->GetSpec().quantization_.Scales();

    if (!vx_tensors.count(idx_bias)) {
        vx_tensors.insert({idx_bias, CreateTvxTensor(graph, tensor_map.at(idx_bias))});
    }
    std::vector<float> scales_bias;
    for (auto it = weight_scales.begin(); it != weight_scales.end(); ++it) {
        scales_bias.push_back(input_scale[0] * (*it));
    }
    std::vector<int32_t> zero_points_bias(scales_bias.size(), 0);
    tim::vx::Quantization quant_bias(quant_weight_type, 0, scales_bias, zero_points_bias);
    auto& bias_spec = vx_tensors[idx_bias]->GetSpec();
    bias_spec.SetQuantization(quant_bias);

    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }
    auto input = vx_tensors[idx_in];
    auto kernel = vx_tensors[idx_kernel];
    auto bias = vx_tensors[idx_bias];
    auto output = vx_tensors[idx_out];

    auto activationCodeScalar = scalar_map.at(idx_act);
    int32_t fuse_code = *reinterpret_cast<const int32_t*>(activationCodeScalar.data.data());

    output = FuseActivation(graph, fuse_code, output);

    auto conv2d = op_creator->Lowering(graph);
    conv2d->BindInput(input);
    conv2d->BindInput(kernel);
    conv2d->BindInput(bias);
    conv2d->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapDepthwiseConv2D(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                       std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                       const TensorMap& tensor_map, const ScalarMap& scalar_map,
                       const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_kernel = inputs[1];
    uint32_t idx_bias = inputs[2];
    uint32_t idx_out = outputs[0];
    uint32_t idx_act;
    inputs.size() == 8 || scalar_map.at(inputs.at(8)).dtype == slang::type::data_type::kBOOL8
            ? idx_act = inputs[7]
            : idx_act = inputs[10];
    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    auto input_scale = vx_tensors[idx_in]->GetSpec().quantization_.Scales();

    if (!vx_tensors.count(idx_kernel)) {  // const/transient weight create here
        vx_tensors.insert({idx_kernel, CreateTvxTensor(graph, tensor_map.at(idx_kernel))});
    }
    tim::vx::QuantType quant_weight_type = vx_tensors[idx_kernel]->GetSpec().quantization_.Type();
    std::vector<float> weight_scales = vx_tensors[idx_kernel]->GetSpec().quantization_.Scales();

    if (!vx_tensors.count(idx_bias)) {
        vx_tensors.insert({idx_bias, CreateTvxTensor(graph, tensor_map.at(idx_bias))});
    }
    std::vector<float> scales_bias;
    for (auto it = weight_scales.begin(); it != weight_scales.end(); ++it) {
        scales_bias.push_back(input_scale[0] * (*it));
    }
    std::vector<int32_t> zero_points_bias(scales_bias.size(), 0);
    tim::vx::Quantization quant_bias(quant_weight_type, 0, scales_bias, zero_points_bias);
    auto& bias_spec = vx_tensors[idx_bias]->GetSpec();
    bias_spec.SetQuantization(quant_bias);

    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }
    auto input = vx_tensors[idx_in];
    auto kernel = vx_tensors[idx_kernel];
    auto bias = vx_tensors[idx_bias];
    auto output = vx_tensors[idx_out];

    const uint8_t* p_act_code = scalar_map.at(idx_act).data.data();
    int32_t fuse_code = *(int32_t*)p_act_code;

    output = FuseActivation(graph, fuse_code, output);

    auto conv2d = op_creator->Lowering(graph);
    conv2d->BindInput(input);
    conv2d->BindInput(kernel);
    conv2d->BindInput(bias);
    conv2d->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapEltwise(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
               std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
               const TensorMap& tensor_map, const ScalarMap& scalar_map,
               const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_in2 = inputs[2];
    uint32_t idx_out = outputs[0];

    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_in1)) {
        vx_tensors.insert({idx_in1, CreateTvxTensor(graph, tensor_map.at(idx_in1))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }
    auto input = vx_tensors[idx_in];
    auto input1 = vx_tensors[idx_in1];
    auto output = vx_tensors[idx_out];

    const uint8_t* fuse_code_data = scalar_map.at(idx_in2).data.data();
    int32_t fuse_code = *reinterpret_cast<const int32_t*>(fuse_code_data);

    output = FuseActivation(graph, fuse_code, output);
    if (output == nullptr) return ANEURALNETWORKS_BAD_DATA;

    auto in_shape = input->GetShape();
    auto in_shape1 = input1->GetShape();
    auto out_shape = output->GetShape();
    if (in_shape < out_shape && input->GetSpec().GetTensorAttribute() != tim::vx::CONSTANT)
        input->GetSpec().SetShape(ExpandedShape(out_shape, in_shape));
    if (in_shape1 < out_shape && input1->GetSpec().GetTensorAttribute() != tim::vx::CONSTANT)
        input1->GetSpec().SetShape(ExpandedShape(out_shape, in_shape1));

    auto eltwise = op_creator->Lowering(graph);
    eltwise->BindInput(input);
    eltwise->BindInput(input1);
    eltwise->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapEltwiseWithNoAct(std::shared_ptr<tim::vx::Graph> graph,
                        std::shared_ptr<OpCreator> op_creator,
                        std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                        const TensorMap& tensor_map, const ScalarMap& scalar_map,
                        const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_out = outputs[0];

    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_in1)) {
        vx_tensors.insert({idx_in1, CreateTvxTensor(graph, tensor_map.at(idx_in1))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }
    auto input = vx_tensors[idx_in];
    auto input1 = vx_tensors[idx_in1];
    auto output = vx_tensors[idx_out];

    auto eltwise_with_no_act = op_creator->Lowering(graph);
    eltwise_with_no_act->BindInput(input);
    eltwise_with_no_act->BindInput(input1);
    eltwise_with_no_act->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapEmbeddingLookup(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                       std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                       const TensorMap& tensor_map, const ScalarMap& scalar_map,
                       const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_lookups = inputs[0];
    uint32_t idx_values = inputs[1];
    uint32_t idx_out = outputs[0];

    if (!vx_tensors.count(idx_lookups)) {
        vx_tensors.insert({idx_lookups, CreateTvxTensor(graph, tensor_map.at(idx_lookups))});
    }
    if (!vx_tensors.count(idx_values)) {
        vx_tensors.insert({idx_values, CreateTvxTensor(graph, tensor_map.at(idx_values))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }  // If not graph input/output, create const or transient tensor

    auto lookups = vx_tensors[idx_lookups];
    auto values = vx_tensors[idx_values];
    auto output = vx_tensors[idx_out];

    auto embedding_lookup = op_creator->Lowering(graph);
    embedding_lookup->BindInput(lookups);
    embedding_lookup->BindInput(values);
    embedding_lookup->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapFullyConnected(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                      std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                      const TensorMap& tensor_map, const ScalarMap& scalar_map,
                      const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_weight = inputs[1];
    uint32_t idx_bias = inputs[2];
    uint32_t idx_act = inputs[3];
    uint32_t idx_out = outputs[0];

    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_weight)) {
        vx_tensors.insert({idx_weight, CreateTvxTensor(graph, tensor_map.at(idx_weight))});
    }
    if (!vx_tensors.count(idx_bias)) {
        vx_tensors.insert({idx_bias, CreateTvxTensor(graph, tensor_map.at(idx_bias))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }  // If not graph input/output, create const or transient tensor

    auto input = vx_tensors[idx_in];
    auto weight = vx_tensors[idx_weight];
    auto bias = vx_tensors[idx_bias];
    auto output = vx_tensors[idx_out];

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

    const int32_t* p_act_code = (int32_t*)scalar_map.at(idx_act).data.data();
    output = FuseActivation(graph, *p_act_code, output);

    auto fc = op_creator->Lowering(graph);
    fc->BindInput(input);
    fc->BindInput(weight);
    fc->BindInput(bias);
    fc->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapGather(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
              std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
              const TensorMap& tensor_map, const ScalarMap& scalar_map,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_indices = inputs[2];
    uint32_t idx_out = outputs[0];

    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_indices)) {
        vx_tensors.insert({idx_indices, CreateTvxTensor(graph, tensor_map.at(idx_indices))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }  // If not graph input/output, create const or transient tensor

    auto input = vx_tensors[idx_in];
    auto indices = vx_tensors[idx_indices];
    auto output = vx_tensors[idx_out];

    auto gather = op_creator->Lowering(graph);
    gather->BindInput(input);
    gather->BindInput(indices);
    gather->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapGroupedConv2d(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                     std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                     const TensorMap& tensor_map, const ScalarMap& scalar_map,
                     const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_kernel = inputs[1];
    uint32_t idx_bias = inputs[2];
    uint32_t idx_out = outputs[0];
    uint32_t idx_act;
    inputs.size() == 9 ? idx_act = inputs[7] : idx_act = inputs[10];
    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    auto input_scale = vx_tensors[idx_in]->GetSpec().quantization_.Scales();

    if (!vx_tensors.count(idx_kernel)) {
        vx_tensors.insert({idx_kernel, CreateTvxTensor(graph, tensor_map.at(idx_kernel))});
    }
    tim::vx::QuantType quant_weight_type = vx_tensors[idx_kernel]->GetSpec().quantization_.Type();
    std::vector<float> weight_scales = vx_tensors[idx_kernel]->GetSpec().quantization_.Scales();

    if (!vx_tensors.count(idx_bias)) {
        vx_tensors.insert({idx_bias, CreateTvxTensor(graph, tensor_map.at(idx_bias))});
    }
    std::vector<float> scales_bias;
    for (auto it = weight_scales.begin(); it != weight_scales.end(); ++it) {
        scales_bias.push_back(input_scale[0] * (*it));
    }
    std::vector<int32_t> zero_points_bias(scales_bias.size(), 0);
    tim::vx::Quantization quant_bias(quant_weight_type, 0, scales_bias, zero_points_bias);
    auto& bias_spec = vx_tensors[idx_bias]->GetSpec();
    bias_spec.SetQuantization(quant_bias);

    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }
    auto input = vx_tensors[idx_in];
    auto kernel = vx_tensors[idx_kernel];
    auto bias = vx_tensors[idx_bias];
    auto output = vx_tensors[idx_out];

    const uint8_t* p_act_code = scalar_map.at(idx_act).data.data();
    int32_t fuse_code = *(int32_t*)p_act_code;

    output = FuseActivation(graph, fuse_code, output);

    auto grouped_conv2d = op_creator->Lowering(graph);
    grouped_conv2d->BindInput(input);
    grouped_conv2d->BindInput(kernel);
    grouped_conv2d->BindInput(bias);
    grouped_conv2d->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapInstanceNormalization(
        std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
        std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
        const TensorMap& tensor_map, const ScalarMap& scalar_map,
        const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_gamma = inputs[1];
    uint32_t idx_beta = inputs[2];
    uint32_t idx_out = outputs[0];
    auto gamma_scalar = scalar_map.at(idx_gamma);
    auto beta_scalar = scalar_map.at(idx_beta);
    tim::vx::TensorSpec spec(tim::vx::DataType::FLOAT32, std::vector<uint32_t>{1},
                             tim::vx::TensorAttribute::CONSTANT);
    if (gamma_scalar.dtype == slang::type::data_type::kFP16) {
        spec.SetDataType(tim::vx::DataType::FLOAT16);
    }
    auto gamma = graph->CreateTensor(spec, gamma_scalar.data.data());
    auto beta = graph->CreateTensor(spec, beta_scalar.data.data());

    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }  // If not graph input/output, create const or transient tensor

    auto input = vx_tensors[idx_in];
    auto output = vx_tensors[idx_out];
    auto instance_normalization = op_creator->Lowering(graph);
    instance_normalization->BindInput(input);
    instance_normalization->BindInput(beta);
    instance_normalization->BindInput(gamma);
    instance_normalization->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapHashtableLookup(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                       std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                       const TensorMap& tensor_map, const ScalarMap& scalar_map,
                       const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_lookups = inputs[0];
    uint32_t idx_keys = inputs[1];
    uint32_t idx_values = inputs[2];
    uint32_t idx_out = outputs[0];
    uint32_t idx_hits = outputs[1];
    if (!vx_tensors.count(idx_lookups)) {
        vx_tensors.insert({idx_lookups, CreateTvxTensor(graph, tensor_map.at(idx_lookups))});
    }
    if (!vx_tensors.count(idx_keys)) {
        vx_tensors.insert({idx_keys, CreateTvxTensor(graph, tensor_map.at(idx_keys))});
    }
    if (!vx_tensors.count(idx_values)) {
        vx_tensors.insert({idx_values, CreateTvxTensor(graph, tensor_map.at(idx_values))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }
    if (!vx_tensors.count(idx_hits)) {
        vx_tensors.insert({idx_hits, CreateTvxTensor(graph, tensor_map.at(idx_hits))});
    }  // If not graph input/output, create const or transient tensor

    auto lookups = vx_tensors[idx_lookups];
    auto keys = vx_tensors[idx_keys];
    auto values = vx_tensors[idx_values];
    auto output = vx_tensors[idx_out];
    auto hits = vx_tensors[idx_hits];
    auto hashtable_lookup = op_creator->Lowering(graph);
    hashtable_lookup->BindInput(lookups);
    hashtable_lookup->BindInput(keys);
    hashtable_lookup->BindInput(values);
    hashtable_lookup->BindOutput(output);
    hashtable_lookup->BindOutput(hits);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapLogicalAndOr(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                    std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                    const TensorMap& tensor_map, const ScalarMap& scalar_map,
                    const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in0 = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_out = outputs[0];

    if (!vx_tensors.count(idx_in0)) {
        vx_tensors.insert({idx_in0, CreateTvxTensor(graph, tensor_map.at(idx_in0))});
    }
    if (!vx_tensors.count(idx_in1)) {
        vx_tensors.insert({idx_in1, CreateTvxTensor(graph, tensor_map.at(idx_in1))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }  // If not graph input/output, create const or transient tensor

    auto input0 = vx_tensors[idx_in0];
    auto input1 = vx_tensors[idx_in1];
    auto output = vx_tensors[idx_out];

    auto logical_and_or = op_creator->Lowering(graph);
    logical_and_or->BindInput(input0);
    logical_and_or->BindInput(input1);
    logical_and_or->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapPack(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
            std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
            const TensorMap& tensor_map, const ScalarMap& scalar_map,
            const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    int32_t inputs_num = inputs.size();
    uint32_t idx_out = outputs[0];
    std::vector<std::shared_ptr<tim::vx::Tensor>> inputs_tensors;
    for (int i = 1; i < inputs_num; ++i) {
        uint32_t idx_in = inputs[i];
        if (!vx_tensors.count(idx_in)) {
            vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
        }
        inputs_tensors.push_back(vx_tensors[idx_in]);
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }

    auto output = vx_tensors[idx_out];

    auto pack = op_creator->Lowering(graph);
    pack->BindInputs(inputs_tensors);
    pack->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapPool2D(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
              std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
              const TensorMap& tensor_map, const ScalarMap& scalar_map,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    uint32_t idx_act;
    (inputs.size() == 7 || inputs.size() == 8) ? idx_act = inputs[6] : idx_act = inputs[9];
    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }
    auto input = vx_tensors[idx_in];
    auto output = vx_tensors[idx_out];
    const uint8_t* p_act_code = scalar_map.at(idx_act).data.data();
    int32_t fuse_code = *(int32_t*)p_act_code;

    output = FuseActivation(graph, fuse_code, output);

    auto pool2d = op_creator->Lowering(graph);
    pool2d->BindInput(input);
    pool2d->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapPrelu(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
             std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
             const TensorMap& tensor_map, const ScalarMap& scalar_map,
             const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_alpha = inputs[1];
    uint32_t idx_out = outputs[0];

    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_alpha)) {
        vx_tensors.insert({idx_alpha, CreateTvxTensor(graph, tensor_map.at(idx_alpha))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }

    auto input = vx_tensors[idx_in];
    auto alpha = vx_tensors[idx_alpha];
    auto alpha_shape = alpha->GetShape();
    bool dims_all_1 = std::all_of(alpha_shape.begin(), alpha_shape.end(),
                                  [](uint32_t dims) { return dims == 1; });
    if (dims_all_1) alpha->GetSpec().SetShape(std::vector<uint32_t>{1});
    auto output = vx_tensors[idx_out];

    auto prelu = op_creator->Lowering(graph);
    prelu->BindInput(input);
    prelu->BindInput(alpha);
    prelu->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapRelationalOp(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                    std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                    const TensorMap& tensor_map, const ScalarMap& scalar_map,
                    const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in0 = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_out = outputs[0];

    if (!vx_tensors.count(idx_in0)) {
        vx_tensors.insert({idx_in0, CreateTvxTensor(graph, tensor_map.at(idx_in0))});
    }
    if (!vx_tensors.count(idx_in1)) {
        vx_tensors.insert({idx_in1, CreateTvxTensor(graph, tensor_map.at(idx_in1))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }  // If not graph input/output, create const or transient tensor

    auto input0 = vx_tensors[idx_in0];
    auto input1 = vx_tensors[idx_in1];
    auto output = vx_tensors[idx_out];

    auto relational_op = op_creator->Lowering(graph);
    relational_op->BindInput(input0);
    relational_op->BindInput(input1);
    relational_op->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapRoi(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
           std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
           const TensorMap& tensor_map, const ScalarMap& scalar_map,
           const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_regions = inputs[1];
    uint32_t idx_batch_index = inputs[2];
    uint32_t idx_out = outputs[0];

    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_regions)) {
        vx_tensors.insert({idx_regions, CreateTvxTensor(graph, tensor_map.at(idx_regions))});
    }
    if (!vx_tensors.count(idx_batch_index)) {
        vx_tensors.insert(
                {idx_batch_index, CreateTvxTensor(graph, tensor_map.at(idx_batch_index))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }  // If not graph input/output, create const or transient tensor

    auto input = vx_tensors[idx_in];
    auto regions = vx_tensors[idx_regions];
    auto batch_index = vx_tensors[idx_batch_index];
    auto output = vx_tensors[idx_out];

    auto roi_align = op_creator->Lowering(graph);
    roi_align->BindInput(input);
    roi_align->BindInput(regions);
    roi_align->BindInput(batch_index);
    roi_align->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapSelect(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
              std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
              const TensorMap& tensor_map, const ScalarMap& scalar_map,
              const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_choose = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_in2 = inputs[2];
    uint32_t idx_out = outputs[0];

    if (!vx_tensors.count(idx_choose)) {
        vx_tensors.insert({idx_choose, CreateTvxTensor(graph, tensor_map.at(idx_choose))});
    }
    if (!vx_tensors.count(idx_in1)) {
        vx_tensors.insert({idx_in1, CreateTvxTensor(graph, tensor_map.at(idx_in1))});
    }
    if (!vx_tensors.count(idx_in2)) {
        vx_tensors.insert({idx_in2, CreateTvxTensor(graph, tensor_map.at(idx_in2))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }  // If not graph input/output, create const or transient tensor

    auto choose = vx_tensors[idx_choose];
    auto input1 = vx_tensors[idx_in1];
    auto input2 = vx_tensors[idx_in2];
    auto output = vx_tensors[idx_out];

    auto select = op_creator->Lowering(graph);
    select->BindInput(choose);
    select->BindInput(input1);
    select->BindInput(input2);
    select->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapSplit(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
             std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
             const TensorMap& tensor_map, const ScalarMap& scalar_map,
             const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_num_splits = inputs[2];
    std::vector<std::shared_ptr<tim::vx::Tensor>> outputs_tensors;
    auto p_num_splits = scalar_map.at(idx_num_splits).data.data();
    int num_splits = *(int32_t*)p_num_splits;
    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    for (int i = 0; i < num_splits; ++i) {
        uint32_t idx_out = outputs[i];
        if (!vx_tensors.count(idx_out)) {
            vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
        }
        outputs_tensors.push_back(vx_tensors[idx_out]);
    }

    auto input = vx_tensors[idx_in];

    auto split = op_creator->Lowering(graph);
    split->BindInput(input);
    split->BindOutputs(outputs_tensors);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapSvdf(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
            std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
            const TensorMap& tensor_map, const ScalarMap& scalar_map,
            const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_weights_feature = inputs[1];
    uint32_t idx_weights_time = inputs[2];
    uint32_t idx_state_out = outputs[0];
    uint32_t idx_out = outputs[1];
    uint32_t idx_bias, idx_state_in, idx_act;
    int32_t fuse_code = 0;
    std::shared_ptr<tim::vx::Tensor> bias;
    if (tensor_map.at(inputs[3]).shape.size() == 1) {
        idx_bias = inputs[3];
        idx_state_in = inputs[4];
        if (!vx_tensors.count(idx_bias)) {
            vx_tensors.insert({idx_bias, CreateTvxTensor(graph, tensor_map.at(idx_bias))});
        }
        bias = vx_tensors[idx_bias];
        if (inputs.size() == 7) {
            idx_act = inputs.back();
            auto p_act = scalar_map.at(idx_act).data.data();
            fuse_code = *(int32_t*)p_act;
        }
    } else {
        idx_state_in = inputs[3];
        if (inputs.size() == 6) {
            idx_act = inputs.back();
            auto p_act = scalar_map.at(idx_act).data.data();
            fuse_code = *(int32_t*)p_act;
        }
    }
    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_weights_feature)) {
        vx_tensors.insert(
                {idx_weights_feature, CreateTvxTensor(graph, tensor_map.at(idx_weights_feature))});
    }
    if (!vx_tensors.count(idx_weights_time)) {
        vx_tensors.insert(
                {idx_weights_time, CreateTvxTensor(graph, tensor_map.at(idx_weights_time))});
    }
    if (!vx_tensors.count(idx_state_in)) {
        vx_tensors.insert({idx_state_in, CreateTvxTensor(graph, tensor_map.at(idx_state_in))});
    }
    if (!vx_tensors.count(idx_state_out)) {
        vx_tensors.insert({idx_state_out, CreateTvxTensor(graph, tensor_map.at(idx_state_out))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }
    auto input = vx_tensors[idx_in];
    auto weights_feature = vx_tensors[idx_weights_feature];
    auto weights_time = vx_tensors[idx_weights_time];
    auto state_in = vx_tensors[idx_state_in];
    auto state_out = vx_tensors[idx_state_out];
    auto output = vx_tensors[idx_out];

    auto svdf = op_creator->Lowering(graph);
    svdf->BindInput(input);
    svdf->BindInput(state_in);
    svdf->BindInput(weights_feature);
    svdf->BindInput(weights_time);
    if (tensor_map.at(inputs[3]).shape.size() == 1) {
        svdf->BindInput(bias);
    }
    svdf->BindOutput(output);
    svdf->BindOutput(state_out);
    return ANEURALNETWORKS_NO_ERROR;
}

int MapTopK(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
            std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
            const TensorMap& tensor_map, const ScalarMap& scalar_map,
            const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    uint32_t idx_indices = outputs[1];
    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }
    if (!vx_tensors.count(idx_indices)) {
        vx_tensors.insert({idx_indices, CreateTvxTensor(graph, tensor_map.at(idx_indices))});
    }
    auto input = vx_tensors[idx_in];
    auto output = vx_tensors[idx_out];
    auto indices = vx_tensors[idx_indices];

    auto top_k = op_creator->Lowering(graph);
    top_k->BindInput(input);
    top_k->BindOutput(output);
    top_k->BindOutput(indices);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapTransposeConv2d(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                       std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                       const TensorMap& tensor_map, const ScalarMap& scalar_map,
                       const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_kernel = inputs[1];
    uint32_t idx_bias = inputs[2];
    uint32_t idx_out = outputs[0];
    uint32_t idx_act;
    inputs.size() == 9 ? idx_act = inputs[7] : idx_act = inputs[9];
    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    auto input_scale = vx_tensors[idx_in]->GetSpec().quantization_.Scales();

    if (!vx_tensors.count(idx_kernel)) {
        vx_tensors.insert({idx_kernel, CreateTvxTensor(graph, tensor_map.at(idx_kernel))});
    }
    tim::vx::QuantType quant_weight_type = vx_tensors[idx_kernel]->GetSpec().quantization_.Type();
    std::vector<float> weight_scales = vx_tensors[idx_kernel]->GetSpec().quantization_.Scales();

    if (!vx_tensors.count(idx_bias)) {
        vx_tensors.insert({idx_bias, CreateTvxTensor(graph, tensor_map.at(idx_bias))});
    }
    std::vector<float> scales_bias;
    for (auto it = weight_scales.begin(); it != weight_scales.end(); ++it) {
        scales_bias.push_back(input_scale[0] * (*it));
    }
    std::vector<int32_t> zero_points_bias(scales_bias.size(), 0);
    tim::vx::Quantization quant_bias(quant_weight_type, 0, scales_bias, zero_points_bias);
    auto& bias_spec = vx_tensors[idx_bias]->GetSpec();
    bias_spec.SetQuantization(quant_bias);

    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }
    auto input = vx_tensors[idx_in];
    auto kernel = vx_tensors[idx_kernel];
    auto bias = vx_tensors[idx_bias];
    auto output = vx_tensors[idx_out];

    const uint8_t* p_act_code = scalar_map.at(idx_act).data.data();
    int32_t fuse_code = *(int32_t*)p_act_code;

    output = FuseActivation(graph, fuse_code, output);

    auto transpose_conv2d = op_creator->Lowering(graph);
    transpose_conv2d->BindInput(input);
    transpose_conv2d->BindInput(kernel);
    transpose_conv2d->BindInput(bias);
    transpose_conv2d->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

}  // namespace sl
}  // namespace android
}  // namespace vsi