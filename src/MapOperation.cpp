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
    const void* data = tensor.data;
    tim::vx::TensorAttribute attr =
            data ? tim::vx::TensorAttribute::CONSTANT : tim::vx::TensorAttribute::TRANSIENT;
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
    return graph->CreateTensor(spec, data);
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
            std::cout << "Unkown fuse code" << std::endl;
            return nullptr;
    }
    auto input = graph->CreateTensor(output->GetSpec().AsTransientSpec());
    op->BindInput(input);
    op->BindOutput(output);
    return input;
}
}  // namespace

int MapActivation(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
               std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
               const TensorMap& tensor_map, const ScalarMap& scalar_map,
               const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
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

    auto activation = op_creator->Lowering(graph);
    activation->BindInput(input);
    activation->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapBatchToSpace(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                     std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                     const TensorMap& tensor_map, const ScalarMap& scalar_map,
                     const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }
    auto input = vx_tensors[idx_in];
    auto output = vx_tensors[idx_out];

    auto space_to_batch = op_creator->Lowering(graph);
    space_to_batch->BindInput(input);
    space_to_batch->BindOutput(output);

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
    for (auto it = weight_scales.begin(); it!=weight_scales.end(); ++it) {
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

int MapDataConvert(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
               std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
               const TensorMap& tensor_map, const ScalarMap& scalar_map,
               const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
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

    auto quantize = op_creator->Lowering(graph);
    quantize->BindInput(input);
    quantize->BindOutput(output);

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
    for (auto it = weight_scales.begin(); it!=weight_scales.end(); ++it) {
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

int MapDepthToSpace(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                     std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                     const TensorMap& tensor_map, const ScalarMap& scalar_map,
                     const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    uint32_t idx_act;
    inputs.size() == 3 ? idx_act = inputs[2] : idx_act = inputs[1];
    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }
    auto input = vx_tensors[idx_in];
    auto output = vx_tensors[idx_out];

    auto depth_to_space = op_creator->Lowering(graph);
    depth_to_space->BindInput(input);
    depth_to_space->BindOutput(output);

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

    auto eltwise = op_creator->Lowering(graph);
    eltwise->BindInput(input);
    eltwise->BindInput(input1);
    eltwise->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapEltwiseUnary(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
               std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
               const TensorMap& tensor_map, const ScalarMap& scalar_map,
               const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
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

    auto simple_op = op_creator->Lowering(graph);
    simple_op->BindInput(input);
    simple_op->BindOutput(output);

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
    for (auto it = weight_scales.begin(); it!=weight_scales.end(); ++it) {
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

int MapL2Normalization(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
               std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
               const TensorMap& tensor_map, const ScalarMap& scalar_map,
               const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
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

    auto l2_normalization = op_creator->Lowering(graph);
    l2_normalization->BindInput(input);
    l2_normalization->BindOutput(output);

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

int MapLogcialNot(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
               std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
               const TensorMap& tensor_map, const ScalarMap& scalar_map,
               const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
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

    auto simple_op = op_creator->Lowering(graph);
    simple_op->BindInput(input);
    simple_op->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapMean(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
               std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
               const TensorMap& tensor_map, const ScalarMap& scalar_map,
               const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
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

    auto mean = op_creator->Lowering(graph);
    mean->BindInput(input);
    mean->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapPad(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
               std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
               const TensorMap& tensor_map, const ScalarMap& scalar_map,
               const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
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

    auto pad = op_creator->Lowering(graph);
    pad->BindInput(input);
    pad->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapPadV2(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
               std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
               const TensorMap& tensor_map, const ScalarMap& scalar_map,
               const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
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

    auto pad2 = op_creator->Lowering(graph);
    pad2->BindInput(input);
    pad2->BindOutput(output);

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

int MapPow(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
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

    auto pow = op_creator->Lowering(graph);
    pow->BindInput(input);
    pow->BindInput(input1);
    pow->BindOutput(output);

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
    auto output = vx_tensors[idx_out];

    auto prelu = op_creator->Lowering(graph);
    prelu->BindInput(input);
    prelu->BindInput(alpha);
    prelu->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapReduce(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
               std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
               const TensorMap& tensor_map, const ScalarMap& scalar_map,
               const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
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

    auto reduce = op_creator->Lowering(graph);
    reduce->BindInput(input);
    reduce->BindOutput(output);

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

int MapReshape(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
               std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
               const TensorMap& tensor_map, const ScalarMap& scalar_map,
               const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
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

    auto reshape = op_creator->Lowering(graph);
    reshape->BindInput(input);
    reshape->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapSoftmax(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
               std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
               const TensorMap& tensor_map, const ScalarMap& scalar_map,
               const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
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

    auto softmax = op_creator->Lowering(graph);
    softmax->BindInput(input);
    softmax->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapSpaceToDepth(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                     std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                     const TensorMap& tensor_map, const ScalarMap& scalar_map,
                     const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    uint32_t idx_act;
    inputs.size() == 3 ? idx_act = inputs[2] : idx_act = inputs[1];
    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }
    auto input = vx_tensors[idx_in];
    auto output = vx_tensors[idx_out];

    auto space_to_depth = op_creator->Lowering(graph);
    space_to_depth->BindInput(input);
    space_to_depth->BindOutput(output);

    return ANEURALNETWORKS_NO_ERROR;
}

int MapSpaceToBatch(std::shared_ptr<tim::vx::Graph> graph, std::shared_ptr<OpCreator> op_creator,
                     std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>>& vx_tensors,
                     const TensorMap& tensor_map, const ScalarMap& scalar_map,
                     const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    if (!vx_tensors.count(idx_in)) {
        vx_tensors.insert({idx_in, CreateTvxTensor(graph, tensor_map.at(idx_in))});
    }
    if (!vx_tensors.count(idx_out)) {
        vx_tensors.insert({idx_out, CreateTvxTensor(graph, tensor_map.at(idx_out))});
    }
    auto input = vx_tensors[idx_in];
    auto output = vx_tensors[idx_out];

    auto space_to_batch = op_creator->Lowering(graph);
    space_to_batch->BindInput(input);
    space_to_batch->BindOutput(output);

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
    for (auto it = weight_scales.begin(); it!=weight_scales.end(); ++it) {
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