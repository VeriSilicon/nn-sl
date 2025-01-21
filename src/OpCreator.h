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

#ifndef VSI_ANDROID_SL_OP_CREATOR_H
#define VSI_ANDROID_SL_OP_CREATOR_H

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>

#include "Utils.h"
#include "slang/functional.h"
#include "slang/type_system.h"
#include "spec/ops/act_with_alpha/spec.h"
#include "spec/ops/activation/spec.h"
#include "spec/ops/arg/spec.h"
#include "spec/ops/batch_matmul/spec.h"
#include "spec/ops/batch_to_space/spec.h"
#include "spec/ops/cast/spec.h"
#include "spec/ops/channel_shuffle/spec.h"
#include "spec/ops/concatenation/spec.h"
#include "spec/ops/conv2d/spec.h"
#include "spec/ops/depth_to_space/spec.h"
#include "spec/ops/depthwise_conv2d/spec.h"
#include "spec/ops/dequantize/spec.h"
#include "spec/ops/eltwise/spec.h"
#include "spec/ops/embedding_lookup/spec.h"
#include "spec/ops/expand_dims/spec.h"
#include "spec/ops/fully_connected/spec.h"
#include "spec/ops/gather/spec.h"
#include "spec/ops/grouped_conv2d/spec.h"
#include "spec/ops/hashtable_lookup/spec.h"
#include "spec/ops/instance_normalization/spec.h"
#include "spec/ops/l2_normalization/spec.h"
#include "spec/ops/local_response_normalization/spec.h"
#include "spec/ops/log_softmax/spec.h"
#include "spec/ops/logical_and_or/spec.h"
#include "spec/ops/logical_not/spec.h"
#include "spec/ops/mean/spec.h"
#include "spec/ops/mirror_pad/spec.h"
#include "spec/ops/pack/spec.h"
#include "spec/ops/pad/spec.h"
#include "spec/ops/pad_v2/spec.h"
#include "spec/ops/pool2d/spec.h"
#include "spec/ops/pow/spec.h"
#include "spec/ops/prelu/spec.h"
#include "spec/ops/quantize/spec.h"
#include "spec/ops/reduce_all_any/spec.h"
#include "spec/ops/reduce_max_min_prod_sum/spec.h"
#include "spec/ops/relational_op/spec.h"
#include "spec/ops/reshape/spec.h"
#include "spec/ops/resize/spec.h"
#include "spec/ops/reverse/spec.h"
#include "spec/ops/roi_align/spec.h"
#include "spec/ops/roi_pooling/spec.h"
#include "spec/ops/select/spec.h"
#include "spec/ops/simple_op/spec.h"
#include "spec/ops/slice/spec.h"
#include "spec/ops/softmax/spec.h"
#include "spec/ops/space_to_batch/spec.h"
#include "spec/ops/space_to_depth/spec.h"
#include "spec/ops/split/spec.h"
#include "spec/ops/squeeze/spec.h"
#include "spec/ops/strided_slice/spec.h"
#include "spec/ops/svdf/spec.h"
#include "spec/ops/tile/spec.h"
#include "spec/ops/topk/spec.h"
#include "spec/ops/transpose/spec.h"
#include "spec/ops/transpose_conv2d/spec.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops.h"

namespace vsi::android::sl {

using TensorMap = std::unordered_map<uint32_t, slang::type::tensor_storage>;
using ScalarMap = std::unordered_map<uint32_t, slang::type::scalar_storage>;

static inline int32_t convertToVxAxis(int32_t axis, uint32_t rank) {
    return static_cast<int32_t>(rank) - (axis < 0 ? rank + axis : axis) - 1;
}

static inline std::vector<uint32_t> convertToVxPerm(std::vector<uint32_t>& perm) {
    uint32_t rank = perm.size();
    std::reverse(perm.begin(), perm.end());
    for (uint32_t i = 0; i < rank; ++i) {
        perm[i] = rank - 1 - perm[i];
    }
    return perm;
}

static inline tim::vx::PadType convertToVxPadType(int32_t code) {
    switch (code) {
        case 0:
            return tim::vx::PadType::AUTO;
        case ANEURALNETWORKS_PADDING_SAME:
            return tim::vx::PadType::SAME;
        case ANEURALNETWORKS_PADDING_VALID:
            return tim::vx::PadType::VALID;
        default:
            LOGW("Padding code: %d is not supported", code);
            return tim::vx::PadType::NONE;
    }
}

static inline tim::vx::DataLayout convertToVxLayout(bool isNCHW) {
    return isNCHW ? tim::vx::DataLayout::WHCN : tim::vx::DataLayout::CWHN;
}

class OpCreator {
   public:
    explicit OpCreator(ANeuralNetworksOperationType type, std::vector<uint32_t> inputs,
                       std::vector<uint32_t> outputs)
        : type_(type), inputs_(std::move(inputs)), outputs_(std::move(outputs)), supported_(true) {}

    virtual ~OpCreator() = default;
    virtual bool checkSupported() = 0;
    virtual std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) = 0;

    [[nodiscard]] ANeuralNetworksOperationType getType() const { return type_; }
    [[nodiscard]] const std::vector<uint32_t>& getInputs() const { return inputs_; }
    [[nodiscard]] const std::vector<uint32_t>& getOutputs() const { return outputs_; }
    [[nodiscard]] bool isSupported() const { return supported_; }

   private:
    ANeuralNetworksOperationType type_;
    std::vector<uint32_t> inputs_;
    std::vector<uint32_t> outputs_;

   protected:
    bool supported_;
};

class PlaceHolderOpCreator final : public OpCreator {
   public:
    explicit PlaceHolderOpCreator(ANeuralNetworksOperationType type) : OpCreator(type, {}, {}) {
        LOGW("OP type: %d is not supported by SL", type);
    }

    bool checkSupported() override { return false; }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Abs>();  // Prevent compiler warnings, not use
    }
};

class AbsCreator final : public OpCreator {
   public:
    AbsCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
               const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_ABS, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("AbsCreator: Invalid number of operands");
            supported_ = false;
        }
        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::simple_op::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::simple_op::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Abs>();
    }

   private:
    op::simple_op::signature signature;
};

class AddCreator final : public OpCreator {
   public:
    AddCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
               const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_ADD, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 1) {
            LOGE("AddCreator: Invalid number of operands");
            supported_ = false;
        }
        uint32_t idx_in = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_act = inputs[2];
        uint32_t idx_out = outputs[0];
        auto in_shape = tensorMap.at(idx_in).shape;
        auto out_shape = tensorMap.at(idx_out).shape;

        bool no_zero_dim =
                std::all_of(in_shape.begin(), in_shape.end(), [](uint32_t dim) { return dim > 0; });
        if (!no_zero_dim) {
            LOGI("AddCreator: Can not support zero Dims before broadcast");
            supported_ = false;
        } else {
            auto dim_iter0 = in_shape.begin();
            auto dim_iter1 = out_shape.begin();
            while (dim_iter0 != in_shape.end() && dim_iter1 != out_shape.end()) {
                if (*dim_iter0 != *dim_iter1) {
                    auto dim_need_broadcast = *dim_iter0 > *dim_iter1 ? *dim_iter1 : *dim_iter0;
                    if (dim_need_broadcast != 1) {
                        LOGE("AddCreator: Invalid shape when broadcast");
                        supported_ = false;
                    }
                }
                ++dim_iter0;
                ++dim_iter1;
            }
        }
        auto act_code_data = scalarMap.at(idx_act).data.data();
        if (act_code_data == nullptr) {
            LOGE("AddCreator: Activation code cannot be null");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::eltwise::Input0(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::eltwise::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::eltwise::Output(tensorMap.at(idx_out));
        std::get<3>(signature.field_tuple) = op::eltwise::Activation(scalarMap.at(idx_act));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Add>();
    }

   private:
    op::eltwise::signature signature;
};

class ArgmaxCreator final : public OpCreator {
   public:
    ArgmaxCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                  const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_ARGMAX, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("ArgmaxCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_axis = inputs[1];
        uint32_t idx_out = outputs[0];
        auto p_axis = scalarMap.at(idx_axis).data.data();
        int32_t axis = *(int32_t*)p_axis;
        uint32_t rank = tensorMap.at(idx_in).shape.size();
        int32_t axis_vx = convertToVxAxis(axis, rank);

        std::get<0>(signature.field_tuple) = op::arg::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::arg::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::arg::Axis(axis_vx);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        int32_t axis = *(int32_t*)p_axis;
        return graph->CreateOperation<tim::vx::ops::ArgMax>(axis);
    }

   private:
    op::arg::signature signature;
};

class ArgminCreator final : public OpCreator {
   public:
    ArgminCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                  const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_ARGMIN, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("ArgminCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_axis = inputs[1];
        uint32_t idx_out = outputs[0];
        auto p_axis = scalarMap.at(idx_axis).data.data();
        int32_t axis = *(int32_t*)p_axis;
        uint32_t rank = tensorMap.at(idx_in).shape.size();
        int32_t axis_vx = convertToVxAxis(axis, rank);

        std::get<0>(signature.field_tuple) = op::arg::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::arg::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::arg::Axis(axis_vx);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        int32_t axis = *(int32_t*)p_axis;
        return graph->CreateOperation<tim::vx::ops::ArgMin>(axis);
    }

   private:
    op::arg::signature signature;
};

class AveragePool2DCreator final : public OpCreator {
   public:
    AveragePool2DCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                         const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_AVERAGE_POOL_2D, inputs, outputs) {
        if ((inputs.size() != 7 && inputs.size() != 8 && inputs.size() != 10 &&
             inputs.size() != 11) ||
            outputs.size() != 1) {
            LOGE("AveragePool2DCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_padding_code, idx_pad_left, idx_pad_right, idx_pad_top, idx_pad_bottom,
                idx_stride_width, idx_stride_height, idx_filter_width, idx_filter_height, idx_act,
                idx_layout;
        uint32_t idx_out = outputs[0];
        std::vector<int32_t> pad = {0, 0, 0, 0};
        std::vector<int32_t> stride = {0, 0};
        std::vector<int32_t> filter = {0, 0};
        int32_t padding_code = 0;
        bool layout = false;  // default to CWHN(false), true implies WHCN.

        if (inputs.size() == 10 || inputs.size() == 11) {
            idx_pad_left = inputs[1];
            idx_pad_right = inputs[2];
            idx_pad_top = inputs[3];
            idx_pad_bottom = inputs[4];
            idx_stride_width = inputs[5];
            idx_stride_height = inputs[6];
            idx_filter_width = inputs[7];
            idx_filter_height = inputs[8];
            idx_act = inputs[9];

            const uint8_t* p_left = scalarMap.at(idx_pad_left).data.data();
            const uint8_t* p_right = scalarMap.at(idx_pad_right).data.data();
            const uint8_t* p_top = scalarMap.at(idx_pad_top).data.data();
            const uint8_t* p_bottom = scalarMap.at(idx_pad_bottom).data.data();
            pad[0] = *(int32_t*)p_left;
            pad[1] = *(int32_t*)p_right;
            pad[2] = *(int32_t*)p_top;
            pad[3] = *(int32_t*)p_bottom;

            if (inputs.size() == 11) {
                idx_layout = inputs[10];
                const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
                layout = *(bool*)p_layout;
            }
        }
        if (inputs.size() == 7 || inputs.size() == 8) {
            idx_padding_code = inputs[1];
            idx_stride_width = inputs[2];
            idx_stride_height = inputs[3];
            idx_filter_width = inputs[4];
            idx_filter_height = inputs[5];
            idx_act = inputs[6];

            const uint8_t* p_code = scalarMap.at(idx_padding_code).data.data();
            padding_code = *(int32_t*)p_code;

            if (inputs.size() == 8) {
                idx_layout = inputs[7];
                const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
                layout = *(bool*)p_layout;
            }
        }
        const uint8_t* p_stride_width = scalarMap.at(idx_stride_width).data.data();
        const uint8_t* p_stride_height = scalarMap.at(idx_stride_height).data.data();
        const uint8_t* p_filter_width = scalarMap.at(idx_filter_width).data.data();
        const uint8_t* p_filter_height = scalarMap.at(idx_filter_height).data.data();
        stride[0] = *(int32_t*)p_stride_width;
        stride[1] = *(int32_t*)p_stride_height;
        filter[0] = *(int32_t*)p_filter_width;
        filter[1] = *(int32_t*)p_filter_height;

        std::get<0>(signature.field_tuple) = op::pool2d::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::pool2d::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::pool2d::Pad(pad);  // construct scalar_feild
        std::get<3>(signature.field_tuple) = op::pool2d::PaddingCode(padding_code);
        std::get<4>(signature.field_tuple) = op::pool2d::Stride(stride);
        std::get<5>(signature.field_tuple) = op::pool2d::Filter(filter);
        std::get<6>(signature.field_tuple) = op::pool2d::Activation(scalarMap.at(idx_act));
        std::get<7>(signature.field_tuple) = op::pool2d::Layout(layout);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_pad = std::get<2>(signature.field_tuple).storage.data.data();
        const uint8_t* p_padding_code = std::get<3>(signature.field_tuple).storage.data.data();
        const uint8_t* p_stride = std::get<4>(signature.field_tuple).storage.data.data();
        const uint8_t* p_filter = std::get<5>(signature.field_tuple).storage.data.data();
        const uint8_t* p_layout = std::get<7>(signature.field_tuple).storage.data.data();
        std::array<uint32_t, 4> pad = {*(uint32_t*)p_pad, *((uint32_t*)p_pad + 1),
                                       *((uint32_t*)p_pad + 2), *((uint32_t*)p_pad + 3)};
        std::array<uint32_t, 2> stride = {*((uint32_t*)p_stride), *((uint32_t*)p_stride + 1)};
        std::array<uint32_t, 2> filter = {*((uint32_t*)p_filter), *((uint32_t*)p_filter + 1)};
        auto layout = convertToVxLayout(*(bool*)p_layout);
        auto pad_type = convertToVxPadType(*(int32_t*)p_padding_code);
        if (pad_type == tim::vx::PadType::AUTO) {
            return graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG_ANDROID, pad,
                                                                filter, stride,
                                                                tim::vx::RoundType::FLOOR, layout);
        } else {
            return graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG_ANDROID,
                                                                pad_type, filter, stride,
                                                                tim::vx::RoundType::FLOOR, layout);
        }
    }

   private:
    op::pool2d::signature signature;
};

class BatchMatmulCreator final : public OpCreator {
   public:
    BatchMatmulCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                       const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_BATCH_MATMUL, inputs, outputs) {
        if ((inputs.size() != 2 && inputs.size() != 4) || outputs.size() != 1) {
            LOGE("BatchMatmulCreator: Invalid number of operands");
            supported_ = false;
        }

        bool adj_x = false, adj_y = false;
        uint32_t idx_in = inputs[0];
        uint32_t idx_in2 = inputs[1];
        uint32_t idx_out = outputs[0];
        if (inputs.size() == 4) {
            uint32_t idx_adj_x = inputs[2];
            uint32_t idx_adj_y = inputs[3];
            auto p_adj_x = scalarMap.at(idx_adj_x).data.data();
            auto p_adj_y = scalarMap.at(idx_adj_y).data.data();
            adj_x = *(bool*)p_adj_x;
            adj_y = *(bool*)p_adj_y;
            if (adj_x && adj_y) {
                LOGI("OpCreator: x and y being true simultaneously is not support in Matmul");
                supported_ = false;
            }
        }
        std::get<0>(signature.field_tuple) = op::batch_matmul::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::batch_matmul::Input2(tensorMap.at(idx_in2));
        std::get<2>(signature.field_tuple) = op::batch_matmul::Output(tensorMap.at(idx_out));
        std::get<3>(signature.field_tuple) = op::batch_matmul::Adj_x(adj_x);
        std::get<4>(signature.field_tuple) = op::batch_matmul::Adj_y(adj_y);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        auto p_adj_x = std::get<3>(signature.field_tuple).storage.data.data();
        auto p_adj_y = std::get<4>(signature.field_tuple).storage.data.data();
        bool adj_x = *(bool*)p_adj_x;
        bool adj_y = *(bool*)p_adj_y;
        return graph->CreateOperation<tim::vx::ops::Matmul>(adj_x, adj_y);
    }

   private:
    op::batch_matmul::signature signature;
};

class BatchToSpaceCreator final : public OpCreator {
   public:
    BatchToSpaceCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                        const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_BATCH_TO_SPACE_ND, inputs, outputs) {
        if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1) {
            LOGE("BatchToSpaceCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_block_size = inputs[1];
        uint32_t idx_layout;
        uint32_t idx_out = outputs[0];

        auto block_size_attr = tensorMap.at(idx_block_size).attr;
        if (block_size_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("BatchToSpaceCreator: Can not support blockSize tensor as INPUT");
            supported_ = false;
        }
        auto block_size_tensor = tensorMap.at(idx_block_size);
        const void* p_block_size = block_size_tensor.data.data();
        const uint32_t block_size_length = block_size_tensor.data.size() / 4;
        std::vector<int32_t> block_size((int32_t*)p_block_size,
                                        (int32_t*)p_block_size + block_size_length);

        bool layout = false;
        if (inputs.size() == 3) {
            idx_layout = inputs[2];
            const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
            layout = *(bool*)p_layout;
        }
        std::get<0>(signature.field_tuple) = op::batch_to_space::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::batch_to_space::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::batch_to_space::BlockSize(block_size);
        std::get<3>(signature.field_tuple) =
                op::batch_to_space::Crop(std::vector<int32_t>{0, 0, 0, 0});
        std::get<4>(signature.field_tuple) = op::batch_to_space::Layout(layout);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_block_size = std::get<2>(signature.field_tuple).storage.data.data();
        const uint8_t* p_layout = std::get<4>(signature.field_tuple).storage.data.data();
        // block_size reverse as input shape reverse
        std::vector<int32_t> block_size = {*((int32_t*)p_block_size + 1), *(int32_t*)p_block_size};
        auto layout = convertToVxLayout(*(bool*)p_layout);
        return graph->CreateOperation<tim::vx::ops::Batch2Space>(
                block_size, std::vector<int32_t>{0, 0, 0, 0}, layout);
    }

   private:
    op::batch_to_space::signature signature;
};

class ConcatenationCreator final : public OpCreator {
   public:
    ConcatenationCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                         const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_CONCATENATION, inputs, outputs) {
        if (inputs.size() < 2 || outputs.size() != 1) {
            LOGE("ConcatenationCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        auto iter = inputs.rbegin();
        uint32_t idx_axis = *iter;
        int32_t input_cnt = inputs.size() - 1;
        uint32_t idx_out = outputs[0];

        auto p_axis = scalarMap.at(idx_axis).data.data();
        int32_t axis = *(int32_t*)p_axis;
        int32_t axis_vx = convertToVxAxis(axis, tensorMap.at(idx_in).shape.size());

        std::get<0>(signature.field_tuple) = op::concatenation::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::concatenation::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::concatenation::Axis(axis_vx);
        std::get<3>(signature.field_tuple) = op::concatenation::Input_cnt(input_cnt);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        const uint8_t* p_input_cnt = std::get<3>(signature.field_tuple).storage.data.data();
        int32_t axis = *(int32_t*)p_axis;
        int32_t input_cnt = *(int32_t*)p_input_cnt;
        return graph->CreateOperation<tim::vx::ops::Concat>(axis, input_cnt);
    }

   private:
    op::concatenation::signature signature;
};

class CastCreator final : public OpCreator {
   public:
    CastCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_CAST, inputs, outputs) {
        if (inputs.size() != 1 && outputs.size() != 1) {
            LOGE("CastCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        auto input_type = tensorMap.at(idx_in).dtype;
        auto quant_type = tensorMap.at(idx_in).qtype;
        if (input_type == slang::type::data_type::kUINT16 &&
            quant_type == slang::type::quant_type::kASYMM) {
            LOGI("CastCreator: Can not support input_dtype uint16 with qtype asymm");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::cast::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::cast::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Cast>();
    }

   private:
    op::cast::signature signature;
};

class ChannelShuffleCreator final : public OpCreator {
   public:
    ChannelShuffleCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                          const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_CHANNEL_SHUFFLE, inputs, outputs) {
        if (inputs.size() != 3 && outputs.size() != 1) {
            LOGE("ChannelShuffleCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_groups = inputs[1];
        uint32_t idx_axis = inputs[2];
        uint32_t idx_out = outputs[0];
        auto p_axis = scalarMap.at(idx_axis).data.data();
        int32_t axis = *(int32_t*)p_axis;
        int32_t axis_vx = convertToVxAxis(axis, tensorMap.at(idx_in).shape.size());

        std::get<0>(signature.field_tuple) = op::channel_shuffle::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::channel_shuffle::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::channel_shuffle::Groups(scalarMap.at(idx_groups));
        std::get<3>(signature.field_tuple) = op::channel_shuffle::Axis(axis_vx);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_groups = std::get<2>(signature.field_tuple).storage.data.data();
        const uint8_t* p_axis = std::get<3>(signature.field_tuple).storage.data.data();
        int32_t axis = *(int32_t*)p_axis;
        int32_t groups = *(int32_t*)p_groups;
        return graph->CreateOperation<tim::vx::ops::ShuffleChannel>(groups, axis);
    }

   private:
    op::channel_shuffle::signature signature;
};

class Conv2DCreator final : public OpCreator {
   public:
    Conv2DCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                  const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_CONV_2D, inputs, outputs) {
        if ((inputs.size() != 7 && inputs.size() != 8 && inputs.size() != 10 &&
             inputs.size() != 11 && inputs.size() != 13) ||
            outputs.size() != 1) {
            LOGE("Conv2DCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_kernel = inputs[1];
        uint32_t idx_bias = inputs[2];
        uint32_t idx_padding_code, idx_pad_left, idx_pad_right, idx_pad_top, idx_pad_bottom,
                idx_stride_width, idx_stride_height, idx_act, idx_dilation_width,
                idx_dilation_height, idx_layout;
        uint32_t idx_out = outputs[0];
        std::vector<int32_t> pad = {0, 0, 0, 0};
        std::vector<int32_t> stride = {0, 0};
        std::vector<int32_t> dilation = {0, 0};
        int32_t padding_code = 0;
        bool layout = false;  // default to CWHN(false), true implies WHCN.

        auto bias_type = tensorMap.at(idx_bias).dtype;
        if (inputs.size() == 7 ||
            scalarMap.at(inputs.at(7)).dtype == slang::type::data_type::kBOOL8) {
            // implies implicit padding
            idx_padding_code = inputs[3];
            idx_stride_width = inputs[4];
            idx_stride_height = inputs[5];
            idx_act = inputs[6];
            const uint8_t* p_code = scalarMap.at(idx_padding_code).data.data();
            padding_code = *(int32_t*)p_code;
            if (inputs.size() == 8 || inputs.size() == 10) {
                idx_layout = inputs[7];
                const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
                layout = *(bool*)p_layout;
                if (inputs.size() == 10) {
                    uint32_t idx_dilation_width = inputs[8];
                    uint32_t idx_dilation_height = inputs[9];
                    const uint8_t* d_width = scalarMap.at(idx_dilation_width).data.data();
                    const uint8_t* d_height = scalarMap.at(idx_dilation_height).data.data();
                    dilation[0] = *(int32_t*)d_width;
                    dilation[1] = *(int32_t*)d_height;
                }
            }
        } else {  // implies explicit padding
            idx_pad_left = inputs[3];
            idx_pad_right = inputs[4];
            idx_pad_top = inputs[5];
            idx_pad_bottom = inputs[6];
            idx_stride_width = inputs[7];
            idx_stride_height = inputs[8];
            idx_act = inputs[9];

            const uint8_t* p_left = scalarMap.at(idx_pad_left).data.data();
            const uint8_t* p_right = scalarMap.at(idx_pad_right).data.data();
            const uint8_t* p_top = scalarMap.at(idx_pad_top).data.data();
            const uint8_t* p_bottom = scalarMap.at(idx_pad_bottom).data.data();
            pad[0] = *(int32_t*)p_left;
            pad[1] = *(int32_t*)p_right;
            pad[2] = *(int32_t*)p_top;
            pad[3] = *(int32_t*)p_bottom;
            if (inputs.size() == 11 || inputs.size() == 13) {
                idx_layout = inputs[10];
                const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
                layout = *(bool*)p_layout;
                if (inputs.size() == 13) {
                    uint32_t idx_dilation_width = inputs[11];
                    uint32_t idx_dilation_height = inputs[12];
                    const uint8_t* d_width = scalarMap.at(idx_dilation_width).data.data();
                    const uint8_t* d_height = scalarMap.at(idx_dilation_height).data.data();
                    dilation[0] = *(int32_t*)d_width;
                    dilation[1] = *(int32_t*)d_height;
                }
            }
        }

        const uint8_t* p_stride_width = scalarMap.at(idx_stride_width).data.data();
        const uint8_t* p_stride_height = scalarMap.at(idx_stride_height).data.data();
        stride[0] = *(int32_t*)p_stride_width;
        stride[1] = *(int32_t*)p_stride_height;

        std::get<0>(signature.field_tuple) = op::conv2d::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::conv2d::Kernel(tensorMap.at(idx_kernel));
        auto kernel_qtype = tensorMap.at(idx_kernel).qtype;
        auto bias = tensorMap.at(idx_bias);
        bias.qtype = kernel_qtype;
        std::get<2>(signature.field_tuple) = op::conv2d::Bias(bias);
        std::get<3>(signature.field_tuple) = op::conv2d::Output(tensorMap.at(idx_out));
        std::get<4>(signature.field_tuple) = op::conv2d::Stride(stride);
        std::get<5>(signature.field_tuple) = op::conv2d::Dilation(dilation);
        std::get<6>(signature.field_tuple) = op::conv2d::PadType(padding_code);
        std::get<7>(signature.field_tuple) = op::conv2d::Pad(pad);  // construct scalar_feild
        std::get<8>(signature.field_tuple) = op::conv2d::Activation(scalarMap.at(idx_act));
        std::get<9>(signature.field_tuple) = op::conv2d::Layout(layout);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint32_t* p_ksize = std::get<1>(signature.field_tuple).storage.shape.data();  // IWHO
        const uint8_t* p_stride = std::get<4>(signature.field_tuple).storage.data.data();
        const uint8_t* p_dilation = std::get<5>(signature.field_tuple).storage.data.data();
        const uint8_t* p_padding_code = std::get<6>(signature.field_tuple).storage.data.data();
        const uint8_t* p_pad = std::get<7>(signature.field_tuple).storage.data.data();
        const uint8_t* p_layout = std::get<9>(signature.field_tuple).storage.data.data();

        std::array<uint32_t, 2> ksize = {*(p_ksize + 1), *(p_ksize + 2)};  // WH
        std::array<uint32_t, 4> pad = {*(uint32_t*)p_pad, *((uint32_t*)p_pad + 1),
                                       *((uint32_t*)p_pad + 2), *((uint32_t*)p_pad + 3)};
        std::array<uint32_t, 2> stride = {*((uint32_t*)p_stride), *((uint32_t*)p_stride + 1)};
        std::array<uint32_t, 2> dilation = {*((uint32_t*)p_dilation), *((uint32_t*)p_dilation + 1)};
        auto pad_type = convertToVxPadType(*(int32_t*)p_padding_code);
        auto layout = convertToVxLayout(*(bool*)p_layout);
        return graph->CreateOperation<tim::vx::ops::Conv2d>(
                0, pad_type, ksize, stride, dilation, pad, 0, layout, tim::vx::DataLayout::IcWHOc);
    }

   private:
    op::conv2d::signature signature;
};

class DepthwiseConv2DCreator final : public OpCreator {
   public:
    DepthwiseConv2DCreator(const std::vector<uint32_t>& inputs,
                           const std::vector<uint32_t>& outputs, const TensorMap& tensorMap,
                           const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_DEPTHWISE_CONV_2D, inputs, outputs) {
        if ((inputs.size() != 8 && inputs.size() != 9 && inputs.size() != 11 &&
             inputs.size() != 12 && inputs.size() != 14) ||
            outputs.size() != 1) {
            LOGE("DepthwiseConv2DCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_kernel = inputs[1];
        uint32_t idx_bias = inputs[2];
        uint32_t idx_padding_code, idx_pad_left, idx_pad_right, idx_pad_top, idx_pad_bottom,
                idx_stride_width, idx_stride_height, idx_multipier, idx_act, idx_dilation_width,
                idx_dilation_height, idx_layout;
        uint32_t idx_out = outputs[0];
        std::vector<int32_t> pad = {0, 0, 0, 0};
        std::vector<int32_t> stride = {0, 0};
        std::vector<int32_t> dilation = {0, 0};
        int32_t padding_code = 0;
        bool layout = false;  // default to CWHN(false), true implies WHCN.

        auto bias_type = tensorMap.at(idx_bias).dtype;
        if (bias_type == slang::type::data_type::kFP16) {
            LOGI("DepthwiseConv2DCreator: Cannot support f16 bias");
            supported_ = false;
        }
        auto kernel_attr = tensorMap.at(idx_bias).attr;
        if (kernel_attr != slang::type::tensor_attr::kCONSTANT) {
            // Fail on target 8mq/8mn/8qxp/8ulp
            LOGI("DepthwiseConv2DCreator: Cannot support non-const weight");
            supported_ = false;
        }
        if (inputs.size() == 8 ||
            scalarMap.at(inputs.at(8)).dtype == slang::type::data_type::kBOOL8) {
            // implies implicit padding
            idx_padding_code = inputs[3];
            idx_stride_width = inputs[4];
            idx_stride_height = inputs[5];
            idx_multipier = inputs[6];
            idx_act = inputs[7];
            const uint8_t* p_code = scalarMap.at(idx_padding_code).data.data();
            padding_code = *(int32_t*)p_code;
            if (inputs.size() == 9 || inputs.size() == 11) {
                idx_layout = inputs[8];
                const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
                layout = *(bool*)p_layout;
                if (inputs.size() == 11) {
                    uint32_t idx_dilation_width = inputs[9];
                    uint32_t idx_dilation_height = inputs[10];
                    const uint8_t* d_width = scalarMap.at(idx_dilation_width).data.data();
                    const uint8_t* d_height = scalarMap.at(idx_dilation_height).data.data();
                    dilation[0] = *(int32_t*)d_width;
                    dilation[1] = *(int32_t*)d_height;
                }
            }
        } else {
            // implies explicit padding
            idx_pad_left = inputs[3];
            idx_pad_right = inputs[4];
            idx_pad_top = inputs[5];
            idx_pad_bottom = inputs[6];
            idx_stride_width = inputs[7];
            idx_stride_height = inputs[8];
            idx_multipier = inputs[9];
            idx_act = inputs[10];

            const uint8_t* p_left = scalarMap.at(idx_pad_left).data.data();
            const uint8_t* p_right = scalarMap.at(idx_pad_right).data.data();
            const uint8_t* p_top = scalarMap.at(idx_pad_top).data.data();
            const uint8_t* p_bottom = scalarMap.at(idx_pad_bottom).data.data();
            pad[0] = *(int32_t*)p_left;
            pad[1] = *(int32_t*)p_right;
            pad[2] = *(int32_t*)p_top;
            pad[3] = *(int32_t*)p_bottom;
            if (inputs.size() == 12 || inputs.size() == 14) {
                idx_layout = inputs[11];
                const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
                layout = *(bool*)p_layout;
                if (inputs.size() == 14) {
                    uint32_t idx_dilation_width = inputs[12];
                    uint32_t idx_dilation_height = inputs[13];
                    const uint8_t* d_width = scalarMap.at(idx_dilation_width).data.data();
                    const uint8_t* d_height = scalarMap.at(idx_dilation_height).data.data();
                    dilation[0] = *(int32_t*)d_width;
                    dilation[1] = *(int32_t*)d_height;
                }
            }
        }

        const uint8_t* p_stride_width = scalarMap.at(idx_stride_width).data.data();
        const uint8_t* p_stride_height = scalarMap.at(idx_stride_height).data.data();
        stride[0] = *(int32_t*)p_stride_width;
        stride[1] = *(int32_t*)p_stride_height;

        auto k_shape = tensorMap.at(idx_kernel).shape;
        if (k_shape[0] != 1) {
            LOGE("DepthwiseConv2DCreator: Invalid kernel shape");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::depthwise_conv2d::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::depthwise_conv2d::Kernel(tensorMap.at(idx_kernel));
        auto kernel_qtype = tensorMap.at(idx_kernel).qtype;
        auto bias = tensorMap.at(idx_bias);
        bias.qtype = kernel_qtype;
        std::get<2>(signature.field_tuple) = op::depthwise_conv2d::Bias(bias);
        std::get<3>(signature.field_tuple) = op::depthwise_conv2d::Output(tensorMap.at(idx_out));
        std::get<4>(signature.field_tuple) = op::depthwise_conv2d::Stride(stride);
        std::get<5>(signature.field_tuple) = op::depthwise_conv2d::Dilation(dilation);
        std::get<6>(signature.field_tuple) = op::depthwise_conv2d::PadType(padding_code);
        std::get<7>(signature.field_tuple) =
                op::depthwise_conv2d::Pad(pad);  // construct scalar_feild
        std::get<8>(signature.field_tuple) =
                op::depthwise_conv2d::Multiplier(scalarMap.at(idx_multipier));
        std::get<9>(signature.field_tuple) =
                op::depthwise_conv2d::Activation(scalarMap.at(idx_act));
        std::get<10>(signature.field_tuple) = op::depthwise_conv2d::Layout(layout);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint32_t* p_ksize = std::get<1>(signature.field_tuple).storage.shape.data();  // OWH1
        const uint8_t* p_stride = std::get<4>(signature.field_tuple).storage.data.data();
        const uint8_t* p_dilation = std::get<5>(signature.field_tuple).storage.data.data();
        const uint8_t* p_padding_code = std::get<6>(signature.field_tuple).storage.data.data();
        const uint8_t* p_pad = std::get<7>(signature.field_tuple).storage.data.data();
        const uint8_t* p_multiplier = std::get<8>(signature.field_tuple).storage.data.data();
        const uint8_t* p_layout = std::get<10>(signature.field_tuple).storage.data.data();

        uint32_t multiplier = *((uint32_t*)p_multiplier);
        std::array<uint32_t, 2> ksize = {*(p_ksize + 1), *(p_ksize + 2)};  // WH
        std::array<uint32_t, 4> pad = {*(uint32_t*)p_pad, *((uint32_t*)p_pad + 1),
                                       *((uint32_t*)p_pad + 2), *((uint32_t*)p_pad + 3)};
        std::array<uint32_t, 2> stride = {*((uint32_t*)p_stride), *((uint32_t*)p_stride + 1)};
        std::array<uint32_t, 2> dilation = {*((uint32_t*)p_dilation), *((uint32_t*)p_dilation + 1)};
        auto pad_type = convertToVxPadType(*(int32_t*)p_padding_code);
        auto layout = convertToVxLayout(*(bool*)p_layout);
        return graph->CreateOperation<tim::vx::ops::Conv2d>(0, pad_type, ksize, stride, dilation,
                                                            pad, multiplier, layout,
                                                            tim::vx::DataLayout::IcWHOc);
    }

   private:
    op::depthwise_conv2d::signature signature;
};

class DepthToSpaceCreator final : public OpCreator {
   public:
    DepthToSpaceCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                        const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_DEPTH_TO_SPACE, inputs, outputs) {
        if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1) {
            LOGE("DepthToSpaceCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_block_size = inputs[1];
        uint32_t idx_layout;
        bool layout = false;
        if (inputs.size() == 3) {
            idx_layout = inputs[2];
            const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
            layout = *(bool*)p_layout;
        }
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::depth_to_space::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::depth_to_space::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) =
                op::depth_to_space::BlockSize(scalarMap.at(idx_block_size));
        std::get<3>(signature.field_tuple) = op::depth_to_space::Layout(layout);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_block_size = std::get<2>(signature.field_tuple).storage.data.data();
        const uint8_t* p_layout = std::get<3>(signature.field_tuple).storage.data.data();
        int32_t block_size = *(int32_t*)p_block_size;
        auto layout = convertToVxLayout(*(bool*)p_layout);
        return graph->CreateOperation<tim::vx::ops::DepthToSpace>(
                block_size, tim::vx::ops::DepthToSpace::DCR_mode, layout);
    }

   private:
    op::depth_to_space::signature signature;
};

class DequantizeCreator final : public OpCreator {
   public:
    DequantizeCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                      const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_DEQUANTIZE, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("DequantizeCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        auto q_type = tensorMap.at(idx_in).qtype;
        if (q_type == slang::type::quant_type::kSYMM_PCQ) {
            LOGI("DequantizeCreator: Cannot support perchannel quantize");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::dequantize::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::dequantize::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::DataConvert>();
    }

   private:
    op::dequantize::signature signature;
};

class DivCreator final : public OpCreator {
   public:
    DivCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
               const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_DIV, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 1) {
            LOGE("DivCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_act = inputs[2];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::eltwise::Input0(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::eltwise::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::eltwise::Output(tensorMap.at(idx_out));
        std::get<3>(signature.field_tuple) = op::eltwise::Activation(scalarMap.at(idx_act));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Div>();
    }

   private:
    op::eltwise::signature signature;
};

class EmbeddingLookupCreator final : public OpCreator {
   public:
    EmbeddingLookupCreator(const std::vector<uint32_t>& inputs,
                           const std::vector<uint32_t>& outputs, const TensorMap& tensorMap,
                           const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_EMBEDDING_LOOKUP, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("EmbeddingLookupCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_lookups = inputs[0];
        uint32_t idx_values = inputs[1];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) =
                op::embedding_lookup::Lookups(tensorMap.at(idx_lookups));
        std::get<1>(signature.field_tuple) = op::embedding_lookup::Values(tensorMap.at(idx_values));
        std::get<2>(signature.field_tuple) = op::embedding_lookup::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::EmbeddingLookup>();
    }

   private:
    op::embedding_lookup::signature signature;
};

class EluCreator final : public OpCreator {
   public:
    EluCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
               const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_ELU, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("EluCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_alpha = inputs[1];
        uint32_t idx_out = outputs[0];
        auto shape = tensorMap.at(idx_in).shape;
        if (shape.size() > 4) {
            LOGE("EluCreator: Elu Only supports up to 4 dimensions");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::act_with_alpha::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::act_with_alpha::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::act_with_alpha::Alpha(scalarMap.at(idx_alpha));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_alpha = std::get<2>(signature.field_tuple).storage.data.data();
        auto datatype = std::get<0>(signature.field_tuple).storage.dtype;
        switch (datatype) {
            case slang::type::data_type::kFP16:
                return graph->CreateOperation<tim::vx::ops::Elu>(*(_Float16*)p_alpha);
            default:
                return graph->CreateOperation<tim::vx::ops::Elu>(*(float*)p_alpha);
        }
    }

   private:
    op::act_with_alpha::signature signature;
};

class EqualCreator final : public OpCreator {
   public:
    EqualCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                 const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_EQUAL, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("EqualCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::relational_op::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::relational_op::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::relational_op::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Equal>();
    }

   private:
    op::relational_op::signature signature;
};

class ExpCreator final : public OpCreator {
   public:
    ExpCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
               const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_EXP, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("ExpCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::simple_op::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::simple_op::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Exp>();
    }

   private:
    op::simple_op::signature signature;
};

class ExpandDimsCreator final : public OpCreator {
   public:
    ExpandDimsCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                      const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_EXPAND_DIMS, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("ExpandDimsCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_axis = inputs[1];
        uint32_t idx_out = outputs[0];
        auto p_axis = scalarMap.at(idx_axis).data.data();
        int32_t axis_android = *(int32_t*)p_axis;
        int32_t rank = tensorMap.at(idx_in).shape.size();
        int32_t axis_vx = convertToVxAxis(axis_android, rank + 1);

        std::get<0>(signature.field_tuple) = op::expand_dims::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::expand_dims::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::expand_dims::Axis(axis_vx);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        auto p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        int32_t axis = *(int32_t*)p_axis;
        int32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
        std::vector<uint32_t> input_shape = std::get<0>(signature.field_tuple).storage.shape;
        std::reverse(input_shape.begin(), input_shape.end());
        std::vector<uint32_t> output_shape(rank + 1, 1);
        for (int i = 0, j = 0; i < output_shape.size(); ++i) {
            if (i != axis) {
                output_shape[i] = input_shape[j];
                ++j;
            }
        }
        return graph->CreateOperation<tim::vx::ops::Reshape>(output_shape);
    }

   private:
    op::expand_dims::signature signature;
};

class FloorCreator final : public OpCreator {
   public:
    FloorCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                 const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_FLOOR, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("FloorCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::simple_op::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::simple_op::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Floor>();
    }

   private:
    op::simple_op::signature signature;
};

class FullyConnectedCreator final : public OpCreator {
   public:
    FullyConnectedCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                          const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_FULLY_CONNECTED, inputs, outputs) {
        if (inputs.size() != 4 || outputs.size() != 1) {
            LOGE("FullyConnectedCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_weight = inputs[1];
        uint32_t idx_bias = inputs[2];
        uint32_t idx_out = outputs[0];

        auto bias_type = tensorMap.at(idx_bias).dtype;
        if (bias_type == slang::type::data_type::kFP16) {
            LOGI("FullyConnectedCreator: Cannot support f16 bias");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::fully_connected::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::fully_connected::Weight(tensorMap.at(idx_weight));
        std::get<2>(signature.field_tuple) = op::fully_connected::Bias(tensorMap.at(idx_bias));
        std::get<3>(signature.field_tuple) = op::fully_connected::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        auto p_weight = (uint32_t*)std::get<1>(signature.field_tuple).storage.shape.data();
        int32_t weight = *(int32_t*)p_weight;
        return graph->CreateOperation<tim::vx::ops::FullyConnected>(0, weight);
    }

   private:
    op::fully_connected::signature signature;
};

class GatherCreator final : public OpCreator {
   public:
    GatherCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                  const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_GATHER, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 1) {
            LOGE("GatherCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_axis = inputs[1];
        uint32_t idx_indices = inputs[2];
        uint32_t idx_out = outputs[0];
        auto p_axis = scalarMap.at(idx_axis).data.data();
        int32_t axis_android = *(int32_t*)p_axis;
        int32_t in_rank = tensorMap.at(idx_in).shape.size();
        int32_t out_rank = tensorMap.at(idx_out).shape.size();
        int32_t axis_vx = convertToVxAxis(axis_android, in_rank);
        if (in_rank > 6 || out_rank > 6) {
            LOGI("GatherCreator: INPUT/OUTPUT rank bigger than 6 is not support");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::gather::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::gather::Indices(tensorMap.at(idx_indices));
        std::get<2>(signature.field_tuple) = op::gather::Output(tensorMap.at(idx_out));
        std::get<3>(signature.field_tuple) = op::gather::Axis(axis_vx);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        auto p_axis = std::get<3>(signature.field_tuple).storage.data.data();
        int32_t axis = *(int32_t*)p_axis;
        return graph->CreateOperation<tim::vx::ops::Gather>(axis, 0);
    }

   private:
    op::gather::signature signature;
};

class GreaterCreator final : public OpCreator {
   public:
    GreaterCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                   const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_GREATER, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("GreaterCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::relational_op::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::relational_op::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::relational_op::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Greater>();
    }

   private:
    op::relational_op::signature signature;
};

class GreaterEqualCreator final : public OpCreator {
   public:
    GreaterEqualCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                        const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_GREATER_EQUAL, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("GreaterCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::relational_op::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::relational_op::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::relational_op::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::GreaterOrEqual>();
    }

   private:
    op::relational_op::signature signature;
};

class GroupedConv2DCreator final : public OpCreator {
   public:
    GroupedConv2DCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                         const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_GROUPED_CONV_2D, inputs, outputs) {
        if ((inputs.size() != 9 && inputs.size() != 12) || outputs.size() != 1) {
            LOGE("GroupedConv2DCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_kernel = inputs[1];
        uint32_t idx_bias = inputs[2];
        uint32_t idx_padding_code, idx_pad_left, idx_pad_right, idx_pad_top, idx_pad_bottom,
                idx_stride_width, idx_stride_height, idx_groups, idx_act, idx_layout;
        uint32_t idx_out = outputs[0];
        std::vector<int32_t> pad = {0, 0, 0, 0};
        std::vector<int32_t> stride = {0, 0};
        std::vector<int32_t> dilation = {0, 0};
        int32_t padding_code = 0;
        bool layout = false;  // default to CWHN(false), true implies WHCN.

        if (inputs.size() == 9) {
            // implies implicit padding
            idx_padding_code = inputs[3];
            idx_stride_width = inputs[4];
            idx_stride_height = inputs[5];
            idx_groups = inputs[6];
            idx_act = inputs[7];
            idx_layout = inputs[8];

            const uint8_t* p_code = scalarMap.at(idx_padding_code).data.data();
            padding_code = *(int32_t*)p_code;
        } else {
            // implies explicit padding
            idx_pad_left = inputs[3];
            idx_pad_right = inputs[4];
            idx_pad_top = inputs[5];
            idx_pad_bottom = inputs[6];
            idx_stride_width = inputs[7];
            idx_stride_height = inputs[8];
            idx_groups = inputs[9];
            idx_act = inputs[10];
            idx_layout = inputs[11];

            const uint8_t* p_left = scalarMap.at(idx_pad_left).data.data();
            const uint8_t* p_right = scalarMap.at(idx_pad_right).data.data();
            const uint8_t* p_top = scalarMap.at(idx_pad_top).data.data();
            const uint8_t* p_bottom = scalarMap.at(idx_pad_bottom).data.data();
            pad[0] = *(int32_t*)p_left;
            pad[1] = *(int32_t*)p_right;
            pad[2] = *(int32_t*)p_top;
            pad[3] = *(int32_t*)p_bottom;
        }
        const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
        layout = *(bool*)p_layout;
        const uint8_t* p_stride_width = scalarMap.at(idx_stride_width).data.data();
        const uint8_t* p_stride_height = scalarMap.at(idx_stride_height).data.data();
        stride[0] = *(int32_t*)p_stride_width;
        stride[1] = *(int32_t*)p_stride_height;
        auto kernel_attr = tensorMap.at(idx_kernel).attr;
        auto bias_attr = tensorMap.at(idx_bias).attr;
        if (bias_attr == slang::type::tensor_attr::kVARIABLE) {
            LOGI("GroupedConv2DCreator: Cannot support non const bias");
            supported_ = false;
        }
        if (stride[0] != stride[1] && kernel_attr == slang::type::tensor_attr::kCONSTANT) {
            LOGI("GroupedConv2DCreator: Cannot support unequal stride when kernel is constant");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::grouped_conv2d::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::grouped_conv2d::Kernel(tensorMap.at(idx_kernel));
        auto kernel_qtype = tensorMap.at(idx_kernel).qtype;
        auto bias = tensorMap.at(idx_bias);
        bias.qtype = kernel_qtype;
        std::get<2>(signature.field_tuple) = op::grouped_conv2d::Bias(bias);
        std::get<3>(signature.field_tuple) = op::grouped_conv2d::Output(tensorMap.at(idx_out));
        std::get<4>(signature.field_tuple) = op::grouped_conv2d::Stride(stride);
        std::get<5>(signature.field_tuple) = op::grouped_conv2d::Dilation(dilation);
        std::get<6>(signature.field_tuple) = op::grouped_conv2d::PadType(padding_code);
        std::get<7>(signature.field_tuple) = op::grouped_conv2d::Pad(pad);
        std::get<8>(signature.field_tuple) = op::grouped_conv2d::Groups(scalarMap.at(idx_groups));
        std::get<9>(signature.field_tuple) = op::grouped_conv2d::Activation(scalarMap.at(idx_act));
        std::get<10>(signature.field_tuple) = op::grouped_conv2d::Layout(layout);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_stride = std::get<4>(signature.field_tuple).storage.data.data();
        const uint8_t* p_padding_code = std::get<6>(signature.field_tuple).storage.data.data();
        const uint8_t* p_pad = std::get<7>(signature.field_tuple).storage.data.data();
        const uint8_t* p_groups = std::get<8>(signature.field_tuple).storage.data.data();
        const uint8_t* p_layout = std::get<10>(signature.field_tuple).storage.data.data();

        uint32_t groups = *((uint32_t*)p_groups);
        std::array<uint32_t, 4> pad = {*(uint32_t*)p_pad, *((uint32_t*)p_pad + 1),
                                       *((uint32_t*)p_pad + 2), *((uint32_t*)p_pad + 3)};
        std::array<uint32_t, 2> stride = {*((uint32_t*)p_stride), *((uint32_t*)p_stride + 1)};
        std::array<uint32_t, 2> dilation = {0, 0};
        auto pad_type = convertToVxPadType(*(int32_t*)p_padding_code);
        auto layout = convertToVxLayout(*(bool*)p_layout);
        if (pad_type != tim::vx::PadType::AUTO) {
            return graph->CreateOperation<tim::vx::ops::GroupedConv2d>(
                    pad_type, stride, dilation, groups, layout, tim::vx::DataLayout::IcWHOc);
        } else {
            return graph->CreateOperation<tim::vx::ops::GroupedConv2d>(
                    pad, stride, dilation, groups, layout, tim::vx::DataLayout::IcWHOc);
        }
    }

   private:
    op::grouped_conv2d::signature signature;
};

class HashtableLookupCreator final : public OpCreator {
   public:
    HashtableLookupCreator(const std::vector<uint32_t>& inputs,
                           const std::vector<uint32_t>& outputs, const TensorMap& tensorMap,
                           const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_HASHTABLE_LOOKUP, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 2) {
            LOGE("HashtableLookupCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_lookups = inputs[0];
        uint32_t idx_keys = inputs[1];
        uint32_t idx_values = inputs[2];
        uint32_t idx_out = outputs[0];
        uint32_t idx_hits = outputs[1];
        std::get<0>(signature.field_tuple) =
                op::hashtable_lookup::Lookups(tensorMap.at(idx_lookups));
        std::get<1>(signature.field_tuple) = op::hashtable_lookup::Keys(tensorMap.at(idx_keys));
        std::get<2>(signature.field_tuple) = op::hashtable_lookup::Values(tensorMap.at(idx_values));
        std::get<3>(signature.field_tuple) = op::hashtable_lookup::Output(tensorMap.at(idx_out));
        std::get<4>(signature.field_tuple) = op::hashtable_lookup::Hits(tensorMap.at(idx_hits));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::HashtableLookup>();
    }

   private:
    op::hashtable_lookup::signature signature;
};

class HardSwishCreator final : public OpCreator {
   public:
    HardSwishCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                     const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_HARD_SWISH, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("HardSwishCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::activation::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::activation::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::HardSwish>();
    }

   private:
    op::activation::signature signature;
};

class InstanceNormalizationCreator final : public OpCreator {
   public:
    InstanceNormalizationCreator(const std::vector<uint32_t>& inputs,
                                 const std::vector<uint32_t>& outputs, const TensorMap& tensorMap,
                                 const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_INSTANCE_NORMALIZATION, inputs, outputs) {
        if (inputs.size() != 5 || outputs.size() != 1) {
            LOGE("InstanceNormalizationCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_gamma = inputs[1];
        uint32_t idx_beta = inputs[2];
        uint32_t idx_epsilon = inputs[3];
        uint32_t idx_layout = inputs[4];
        uint32_t idx_out = outputs[0];
        auto gamma_type = scalarMap.at(inputs[1]).dtype;
        auto beta_type = scalarMap.at(inputs[2]).dtype;

        std::get<0>(signature.field_tuple) =
                op::instance_normalization::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) =
                op::instance_normalization::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) =
                op::instance_normalization::Epsilon(scalarMap.at(idx_epsilon));
        std::get<3>(signature.field_tuple) =
                op::instance_normalization::Layout(scalarMap.at(idx_layout));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_epsilon = std::get<2>(signature.field_tuple).storage.data.data();
        const uint8_t* p_layout = std::get<3>(signature.field_tuple).storage.data.data();
        auto input_type = std::get<0>(signature.field_tuple).storage.dtype;
        auto layout = convertToVxLayout(*(bool*)p_layout);
        layout_ = layout;
        if (input_type == slang::type::data_type::kFP16) {
            return graph->CreateOperation<tim::vx::ops::InstanceNormalization>(
                    *(_Float16*)p_epsilon, layout);
        } else {
            return graph->CreateOperation<tim::vx::ops::InstanceNormalization>(*(float*)p_epsilon,
                                                                               layout);
        }
    }

    [[nodiscard]] tim::vx::DataLayout getLayout() const { return layout_; }

   private:
    tim::vx::DataLayout layout_;
    op::instance_normalization::signature signature;
};

class L2NormalizationCreator final : public OpCreator {
   public:
    L2NormalizationCreator(const std::vector<uint32_t>& inputs,
                           const std::vector<uint32_t>& outputs, const TensorMap& tensorMap,
                           const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_L2_NORMALIZATION, inputs, outputs) {
        if ((inputs.size() != 1 && inputs.size() != 2) || outputs.size() != 1) {
            LOGE("L2NormalizationCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];

        int32_t axis = -1;
        if (inputs.size() == 2) {
            uint32_t idx_axis = inputs[1];
            auto p_axis = scalarMap.at(idx_axis).data.data();
            axis = *(int32_t*)p_axis;
        }
        uint32_t rank = tensorMap.at(idx_in).shape.size();
        int32_t axis_vx = convertToVxAxis(axis, rank);
        std::get<0>(signature.field_tuple) = op::l2_normalization::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::l2_normalization::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::l2_normalization::Axis(axis_vx);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        int32_t axis = *(int32_t*)p_axis;
        return graph->CreateOperation<tim::vx::ops::L2Normalization>(axis);
    }

   private:
    op::l2_normalization::signature signature;
};

class LessCreator final : public OpCreator {
   public:
    LessCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_LESS, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("LessCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::relational_op::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::relational_op::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::relational_op::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Less>();
    }

   private:
    op::relational_op::signature signature;
};

class LessEqualCreator final : public OpCreator {
   public:
    LessEqualCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                     const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_LESS_EQUAL, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("LessEqualCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::relational_op::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::relational_op::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::relational_op::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::LessOrEqual>();
    }

   private:
    op::relational_op::signature signature;
};

class LocalResponseNormalizationCreator final : public OpCreator {
   public:
    LocalResponseNormalizationCreator(const std::vector<uint32_t>& inputs,
                                      const std::vector<uint32_t>& outputs,
                                      const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION, inputs, outputs) {
        if ((inputs.size() != 5 && inputs.size() != 6) || outputs.size() != 1) {
            LOGE("LocalResponseNormalizationCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_radius = inputs[1];
        uint32_t idx_bias = inputs[2];
        uint32_t idx_alpha = inputs[3];
        uint32_t idx_beta = inputs[4];
        uint32_t idx_out = outputs[0];

        int32_t axis_android = -1;
        if (inputs.size() == 6) {
            uint32_t idx_axis = inputs[5];
            auto p_axis = scalarMap.at(idx_axis).data.data();
            axis_android = *(int32_t*)p_axis;
        }
        int32_t axis_vx = convertToVxAxis(axis_android, tensorMap.at(idx_in).shape.size());
        std::get<0>(signature.field_tuple) =
                op::local_response_normalization::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) =
                op::local_response_normalization::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) =
                op::local_response_normalization::Radius(scalarMap.at(idx_radius));
        std::get<3>(signature.field_tuple) =
                op::local_response_normalization::Bias(scalarMap.at(idx_bias));
        std::get<4>(signature.field_tuple) =
                op::local_response_normalization::Alpha(scalarMap.at(idx_alpha));
        std::get<5>(signature.field_tuple) =
                op::local_response_normalization::Beta(scalarMap.at(idx_beta));
        std::get<6>(signature.field_tuple) = op::local_response_normalization::Axis(axis_vx);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_radius = std::get<2>(signature.field_tuple).storage.data.data();
        const uint8_t* p_bias = std::get<3>(signature.field_tuple).storage.data.data();
        const uint8_t* p_alpha = std::get<4>(signature.field_tuple).storage.data.data();
        const uint8_t* p_beta = std::get<5>(signature.field_tuple).storage.data.data();
        const uint8_t* p_axis = std::get<6>(signature.field_tuple).storage.data.data();
        uint32_t size = *(int32_t*)p_radius * 2 + 1;
        auto input_type = std::get<0>(signature.field_tuple).storage.dtype;
        if (input_type == slang::type::data_type::kFP32) {
            return graph->CreateOperation<tim::vx::ops::LocalResponseNormalization>(
                    size, *(float*)p_alpha, *(float*)p_beta, *(float*)p_bias, *(int32_t*)p_axis);
        } else {
            return graph->CreateOperation<tim::vx::ops::LocalResponseNormalization>(
                    size, *(_Float16*)p_alpha, *(_Float16*)p_beta, *(_Float16*)p_bias,
                    *(int32_t*)p_axis);
        }
    }

   private:
    op::local_response_normalization::signature signature;
};

class LogCreator final : public OpCreator {
   public:
    LogCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
               const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_LOG, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("LogCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        auto attr = tensorMap.at(idx_in).dtype;
        if (attr == slang::type::data_type::kFP16) {
            LOGI("LogCreator: Cannot support f16 input");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::simple_op::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::simple_op::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Log>();
    }

   private:
    op::simple_op::signature signature;
};

class LogisticCreator final : public OpCreator {
   public:
    LogisticCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                    const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_LOGISTIC, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("LogisticCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        auto shape = tensorMap.at(idx_in).shape;
        if (shape.size() > 4) {
            LOGE("LogisticCreator: Logistic Only supports up to 4 dimensions");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::activation::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::activation::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Sigmoid>();
    }

   private:
    op::activation::signature signature;
};

class LogicalAndCreator final : public OpCreator {
   public:
    LogicalAndCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                      const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_LOGICAL_AND, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("LogicalAndCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::logical_and_or::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::logical_and_or::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::logical_and_or::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::LogicalAnd>();
    }

   private:
    op::logical_and_or::signature signature;
};

class LogicalNotCreator final : public OpCreator {
   public:
    LogicalNotCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                      const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_LOGICAL_NOT, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("LogicalNotCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::logical_not::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::logical_not::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::LogicalNot>();
    }

   private:
    op::logical_not::signature signature;
};

class LogicalOrCreator final : public OpCreator {
   public:
    LogicalOrCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                     const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_LOGICAL_OR, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("LogicalOrCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::logical_and_or::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::logical_and_or::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::logical_and_or::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::LogicalOr>();
    }

   private:
    op::logical_and_or::signature signature;
};

class LogSoftmaxCreator final : public OpCreator {
   public:
    LogSoftmaxCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                      const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_LOG_SOFTMAX, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 1) {
            LOGE("LogSoftmaxCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_beta = inputs[1];
        uint32_t idx_axis = inputs[2];
        uint32_t idx_out = outputs[0];

        std::get<0>(signature.field_tuple) = op::log_softmax::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::log_softmax::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::log_softmax::Beta(scalarMap.at(idx_beta));
        std::get<3>(signature.field_tuple) = op::log_softmax::Axis(scalarMap.at(idx_axis));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
        auto datatype = std::get<2>(signature.field_tuple).storage.dtype;
        const uint8_t* p_beta = std::get<2>(signature.field_tuple).storage.data.data();
        const uint8_t* p_axis = std::get<3>(signature.field_tuple).storage.data.data();
        int32_t axis_android = *(int32_t*)p_axis;
        int32_t axis_vx = convertToVxAxis(axis_android, rank);
        if (datatype == slang::type::data_type::kFP16) {
            auto beta = *(_Float16*)p_beta;
            return graph->CreateOperation<tim::vx::ops::LogSoftmax>(axis_vx, beta);
        } else {
            auto beta = *(float*)p_beta;
            return graph->CreateOperation<tim::vx::ops::LogSoftmax>(axis_vx, beta);
        }
    }

   private:
    op::log_softmax::signature signature;
};

class L2Pool2DCreator final : public OpCreator {
   public:
    L2Pool2DCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                    const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_L2_POOL_2D, inputs, outputs) {
        if ((inputs.size() != 7 && inputs.size() != 8 && inputs.size() != 10 &&
             inputs.size() != 11) ||
            outputs.size() != 1) {
            LOGE("L2Pool2DCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_padding_code, idx_pad_left, idx_pad_right, idx_pad_top, idx_pad_bottom,
                idx_stride_width, idx_stride_height, idx_filter_width, idx_filter_height, idx_act,
                idx_layout;
        uint32_t idx_out = outputs[0];
        std::vector<int32_t> pad = {0, 0, 0, 0};
        std::vector<int32_t> stride = {0, 0};
        std::vector<int32_t> filter = {0, 0};
        int32_t padding_code = 0;
        bool layout = false;  // default to CWHN(false), true implies WHCN.

        if (inputs.size() == 10 || inputs.size() == 11) {
            idx_pad_left = inputs[1];
            idx_pad_right = inputs[2];
            idx_pad_top = inputs[3];
            idx_pad_bottom = inputs[4];
            idx_stride_width = inputs[5];
            idx_stride_height = inputs[6];
            idx_filter_width = inputs[7];
            idx_filter_height = inputs[8];
            idx_act = inputs[9];

            const uint8_t* p_left = scalarMap.at(idx_pad_left).data.data();
            const uint8_t* p_right = scalarMap.at(idx_pad_right).data.data();
            const uint8_t* p_top = scalarMap.at(idx_pad_top).data.data();
            const uint8_t* p_bottom = scalarMap.at(idx_pad_bottom).data.data();
            pad[0] = *(int32_t*)p_left;
            pad[1] = *(int32_t*)p_right;
            pad[2] = *(int32_t*)p_top;
            pad[3] = *(int32_t*)p_bottom;

            if (inputs.size() == 11) {
                idx_layout = inputs[10];
                const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
                layout = *(bool*)p_layout;
            }
        }
        if (inputs.size() == 7 || inputs.size() == 8) {
            idx_padding_code = inputs[1];
            idx_stride_width = inputs[2];
            idx_stride_height = inputs[3];
            idx_filter_width = inputs[4];
            idx_filter_height = inputs[5];
            idx_act = inputs[6];

            const uint8_t* p_code = scalarMap.at(idx_padding_code).data.data();
            padding_code = *(int32_t*)p_code;

            if (inputs.size() == 8) {
                idx_layout = inputs[7];
                const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
                layout = *(bool*)p_layout;
            }
        }
        const uint8_t* p_stride_width = scalarMap.at(idx_stride_width).data.data();
        const uint8_t* p_stride_height = scalarMap.at(idx_stride_height).data.data();
        const uint8_t* p_filter_width = scalarMap.at(idx_filter_width).data.data();
        const uint8_t* p_filter_height = scalarMap.at(idx_filter_height).data.data();
        stride[0] = *(int32_t*)p_stride_width;
        stride[1] = *(int32_t*)p_stride_height;
        filter[0] = *(int32_t*)p_filter_width;
        filter[1] = *(int32_t*)p_filter_height;

        std::get<0>(signature.field_tuple) = op::pool2d::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::pool2d::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::pool2d::Pad(pad);  // construct scalar_feild
        std::get<3>(signature.field_tuple) = op::pool2d::PaddingCode(padding_code);
        std::get<4>(signature.field_tuple) = op::pool2d::Stride(stride);
        std::get<5>(signature.field_tuple) = op::pool2d::Filter(filter);
        std::get<6>(signature.field_tuple) = op::pool2d::Activation(scalarMap.at(idx_act));
        std::get<7>(signature.field_tuple) = op::pool2d::Layout(layout);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_pad = std::get<2>(signature.field_tuple).storage.data.data();
        const uint8_t* p_padding_code = std::get<3>(signature.field_tuple).storage.data.data();
        const uint8_t* p_stride = std::get<4>(signature.field_tuple).storage.data.data();
        const uint8_t* p_filter = std::get<5>(signature.field_tuple).storage.data.data();
        const uint8_t* p_layout = std::get<7>(signature.field_tuple).storage.data.data();
        std::array<uint32_t, 4> pad = {*(uint32_t*)p_pad, *((uint32_t*)p_pad + 1),
                                       *((uint32_t*)p_pad + 2), *((uint32_t*)p_pad + 3)};
        std::array<uint32_t, 2> stride = {*((uint32_t*)p_stride), *((uint32_t*)p_stride + 1)};
        std::array<uint32_t, 2> filter = {*((uint32_t*)p_filter), *((uint32_t*)p_filter + 1)};
        auto pad_type = convertToVxPadType(*(int32_t*)p_padding_code);
        auto layout = convertToVxLayout(*(bool*)p_layout);
        if (pad_type == tim::vx::PadType::AUTO) {
            return graph->CreateOperation<tim::vx::ops::Pool2d>(
                    tim::vx::PoolType::L2, pad, filter, stride, tim::vx::RoundType::FLOOR, layout);
        } else {
            return graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::L2, pad_type,
                                                                filter, stride,
                                                                tim::vx::RoundType::FLOOR, layout);
        }
    }

   private:
    op::pool2d::signature signature;
};

class MaxPool2DCreator final : public OpCreator {
   public:
    MaxPool2DCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                     const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_MAX_POOL_2D, inputs, outputs) {
        if ((inputs.size() != 7 && inputs.size() != 8 && inputs.size() != 10 &&
             inputs.size() != 11) ||
            outputs.size() != 1) {
            LOGE("MaxPool2DCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_padding_code, idx_pad_left, idx_pad_right, idx_pad_top, idx_pad_bottom,
                idx_stride_width, idx_stride_height, idx_filter_width, idx_filter_height, idx_act,
                idx_layout;
        uint32_t idx_out = outputs[0];
        std::vector<int32_t> pad = {0, 0, 0, 0};
        std::vector<int32_t> stride = {0, 0};
        std::vector<int32_t> filter = {0, 0};
        int32_t padding_code = 0;
        bool layout = false;  // default to CWHN(false), true implies WHCN.

        if (inputs.size() == 10 || inputs.size() == 11) {
            idx_pad_left = inputs[1];
            idx_pad_right = inputs[2];
            idx_pad_top = inputs[3];
            idx_pad_bottom = inputs[4];
            idx_stride_width = inputs[5];
            idx_stride_height = inputs[6];
            idx_filter_width = inputs[7];
            idx_filter_height = inputs[8];
            idx_act = inputs[9];

            const uint8_t* p_left = scalarMap.at(idx_pad_left).data.data();
            const uint8_t* p_right = scalarMap.at(idx_pad_right).data.data();
            const uint8_t* p_top = scalarMap.at(idx_pad_top).data.data();
            const uint8_t* p_bottom = scalarMap.at(idx_pad_bottom).data.data();
            pad[0] = *(int32_t*)p_left;
            pad[1] = *(int32_t*)p_right;
            pad[2] = *(int32_t*)p_top;
            pad[3] = *(int32_t*)p_bottom;

            if (inputs.size() == 11) {
                idx_layout = inputs[10];
                const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
                layout = *(bool*)p_layout;
            }
        }
        if (inputs.size() == 7 || inputs.size() == 8) {
            idx_padding_code = inputs[1];
            idx_stride_width = inputs[2];
            idx_stride_height = inputs[3];
            idx_filter_width = inputs[4];
            idx_filter_height = inputs[5];
            idx_act = inputs[6];

            const uint8_t* p_code = scalarMap.at(idx_padding_code).data.data();
            padding_code = *(int32_t*)p_code;

            if (inputs.size() == 8) {
                idx_layout = inputs[7];
                const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
                layout = *(bool*)p_layout;
            }
        }
        const uint8_t* p_stride_width = scalarMap.at(idx_stride_width).data.data();
        const uint8_t* p_stride_height = scalarMap.at(idx_stride_height).data.data();
        const uint8_t* p_filter_width = scalarMap.at(idx_filter_width).data.data();
        const uint8_t* p_filter_height = scalarMap.at(idx_filter_height).data.data();
        stride[0] = *(int32_t*)p_stride_width;
        stride[1] = *(int32_t*)p_stride_height;
        filter[0] = *(int32_t*)p_filter_width;
        filter[1] = *(int32_t*)p_filter_height;

        std::get<0>(signature.field_tuple) = op::pool2d::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::pool2d::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::pool2d::Pad(pad);  // construct scalar_feild
        std::get<3>(signature.field_tuple) = op::pool2d::PaddingCode(padding_code);
        std::get<4>(signature.field_tuple) = op::pool2d::Stride(stride);
        std::get<5>(signature.field_tuple) = op::pool2d::Filter(filter);
        std::get<6>(signature.field_tuple) = op::pool2d::Activation(scalarMap.at(idx_act));
        std::get<7>(signature.field_tuple) = op::pool2d::Layout(layout);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_pad = std::get<2>(signature.field_tuple).storage.data.data();
        const uint8_t* p_padding_code = std::get<3>(signature.field_tuple).storage.data.data();
        const uint8_t* p_stride = std::get<4>(signature.field_tuple).storage.data.data();
        const uint8_t* p_filter = std::get<5>(signature.field_tuple).storage.data.data();
        const uint8_t* p_layout = std::get<7>(signature.field_tuple).storage.data.data();
        std::array<uint32_t, 4> pad = {*(uint32_t*)p_pad, *((uint32_t*)p_pad + 1),
                                       *((uint32_t*)p_pad + 2), *((uint32_t*)p_pad + 3)};
        std::array<uint32_t, 2> stride = {*((uint32_t*)p_stride), *((uint32_t*)p_stride + 1)};
        std::array<uint32_t, 2> filter = {*((uint32_t*)p_filter), *((uint32_t*)p_filter + 1)};

        auto pad_type = convertToVxPadType(*(int32_t*)p_padding_code);
        auto layout = convertToVxLayout(*(bool*)p_layout);
        if (pad_type == tim::vx::PadType::AUTO) {
            return graph->CreateOperation<tim::vx::ops::Pool2d>(
                    tim::vx::PoolType::MAX, pad, filter, stride, tim::vx::RoundType::FLOOR, layout);
        } else {
            return graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::MAX, pad_type,
                                                                filter, stride,
                                                                tim::vx::RoundType::FLOOR, layout);
        }
    }

   private:
    op::pool2d::signature signature;
};

class MaximumCreator final : public OpCreator {
   public:
    MaximumCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                   const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_MAXIMUM, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("MaximumCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::eltwise::Input0(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::eltwise::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::eltwise::Output(tensorMap.at(idx_out));
        std::get<3>(signature.field_tuple) = op::eltwise::Activation(0);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Maximum>();
    }

   private:
    op::eltwise::signature signature;
};

class MeanCreator final : public OpCreator {
   public:
    MeanCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_MEAN, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 1) {
            LOGE("MeanCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_axis = inputs[1];
        uint32_t idx_keepdims = inputs[2];
        uint32_t idx_out = outputs[0];

        auto axis_attr = tensorMap.at(idx_axis).attr;
        if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("MeanCreator: Cannot support axis tensor as INPUT");
            supported_ = false;
        }
        auto p_keepdims = (bool*)scalarMap.at(idx_keepdims).data.data();
        bool keepdims = *p_keepdims;

        std::get<0>(signature.field_tuple) = op::mean::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::mean::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::mean::Axis(tensorMap.at(idx_axis));
        std::get<3>(signature.field_tuple) = op::mean::KeepDims(keepdims);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        std::vector<int32_t> axis_vx;
        const uint32_t in_rank = std::get<0>(signature.field_tuple).storage.shape.size();
        const uint32_t out_rank = std::get<1>(signature.field_tuple).storage.shape.size();
        const void* p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        const uint32_t axis_length = std::get<2>(signature.field_tuple).storage.data.size() / 4;
        for (int i = 0; i < axis_length; i++) {
            int32_t axis_android = *((int32_t*)p_axis + i);
            axis_vx.push_back(convertToVxAxis(axis_android, in_rank));
        }
        const uint8_t* p_keepdims = std::get<3>(signature.field_tuple).storage.data.data();
        bool keepdims = *(bool*)p_keepdims;
        if (in_rank == out_rank) {
            keepdims = true;
        }
        return graph->CreateOperation<tim::vx::ops::ReduceMean>(axis_vx, keepdims);
    }

   private:
    op::mean::signature signature;
};

class MinimumCreator final : public OpCreator {
   public:
    MinimumCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                   const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_MINIMUM, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("MinimumCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::eltwise::Input0(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::eltwise::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::eltwise::Output(tensorMap.at(idx_out));
        std::get<3>(signature.field_tuple) = op::eltwise::Activation(0);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Minimum>();
    }

   private:
    op::eltwise::signature signature;
};

class MirrorPadCreator final : public OpCreator {
   public:
    MirrorPadCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                     const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_MIRROR_PAD, inputs, outputs) {
        if ((inputs.size() != 3) || outputs.size() != 1) {
            LOGE("MirrorPadCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_pad = inputs[1];
        uint32_t idx_mode = inputs[2];
        uint32_t idx_out = outputs[0];

        auto rank = tensorMap.at(idx_in).shape.size();
        if (rank > 6) {
            LOGI("MirrorPadCreator: Cannot support INPUT rank more than 6 in MirrorPad");
            supported_ = false;
        }
        auto pad_attr = tensorMap.at(idx_pad).attr;
        if (pad_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("MirrorPadCreator: Cannot support Pad tensor as INPUT in MirrorPad");
            supported_ = false;
        }
        auto p_pad = (int32_t*)tensorMap.at(idx_pad).data.data();
        uint32_t pad_length = tensorMap.at(idx_pad).data.size() / 4;
        std::vector<int32_t> pad(p_pad, p_pad + pad_length);

        std::get<0>(signature.field_tuple) = op::mirror_pad::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::mirror_pad::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::mirror_pad::Pad(pad);
        std::get<3>(signature.field_tuple) = op::mirror_pad::PadMode(scalarMap.at(idx_mode));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
        auto p_pad = (uint32_t*)std::get<2>(signature.field_tuple).storage.data.data();
        auto p_pad_mode = (int32_t*)std::get<3>(signature.field_tuple).storage.data.data();
        std::vector<uint32_t> front_size, back_size;
        for (int i = 0; i < rank; ++i) {
            front_size.push_back(*(p_pad + i * 2));
            back_size.push_back(*(p_pad + i * 2 + 1));
        }
        // The dim value reverses along with the shape value.
        std::reverse(front_size.begin(), front_size.end());
        std::reverse(back_size.begin(), back_size.end());
        int32_t pad_mode = *p_pad_mode;
        auto vsi_pad_mode = tim::vx::ops::Pad::PAD_MODE_CONSTANT;
        switch (pad_mode) {
            case 0:
                vsi_pad_mode = tim::vx::ops::Pad::PAD_MODE_REFLECT;
                break;
            case 1:
                vsi_pad_mode = tim::vx::ops::Pad::PAD_MODE_SYMMETRIC;
                break;
            default:
                LOGE("MirrorPadCreator:: Invalid pad mode");
                break;
        }
        return graph->CreateOperation<tim::vx::ops::Pad>(front_size, back_size, 0, vsi_pad_mode);
    }

   private:
    op::mirror_pad::signature signature;
};

class MulCreator final : public OpCreator {
   public:
    MulCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
               const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_MUL, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 1) {
            LOGE("MulCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_act = inputs[2];
        uint32_t idx_out = outputs[0];
        if (tensorMap.at(idx_in).dtype == slang::type::data_type::kINT32) {
            LOGI("MulCreator: Cannot support int32 INPUT");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::eltwise::Input0(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::eltwise::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::eltwise::Output(tensorMap.at(idx_out));
        std::get<3>(signature.field_tuple) = op::eltwise::Activation(scalarMap.at(idx_act));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Multiply>();
    }

   private:
    op::eltwise::signature signature;
};

class NegCreator final : public OpCreator {
   public:
    NegCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
               const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_NEG, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("NegCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        auto attr = tensorMap.at(idx_in).dtype;
        if (attr == slang::type::data_type::kFP16) {
            LOGI("NegCreator: Cannot support f16 input");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::simple_op::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::simple_op::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Neg>();
    }

   private:
    op::simple_op::signature signature;
};

class NotEqualCreator final : public OpCreator {
   public:
    NotEqualCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                    const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_NOT_EQUAL, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("NotEqualCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::relational_op::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::relational_op::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::relational_op::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::NotEqual>();
    }

   private:
    op::relational_op::signature signature;
};

class PackCreator final : public OpCreator {
   public:
    PackCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_PACK, inputs, outputs) {
        if ((inputs.size() < 2) || outputs.size() != 1) {
            LOGE("PackCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_axis = inputs[0];
        uint32_t idx_in = inputs[1];
        uint32_t idx_out = outputs[0];
        int32_t input_cnt = inputs.size() - 1;
        auto p_axis = scalarMap.at(idx_axis).data.data();
        int32_t axis_android = *(int32_t*)p_axis;
        int32_t axis_vx = convertToVxAxis(axis_android, tensorMap.at(idx_out).shape.size());
        std::get<0>(signature.field_tuple) = op::pack::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::pack::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::pack::Axis(axis_vx);
        std::get<3>(signature.field_tuple) = op::pack::Input_cnt(input_cnt);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        auto p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        auto p_input_cnt = std::get<3>(signature.field_tuple).storage.data.data();
        int32_t axis = *(int32_t*)p_axis;
        int32_t input_cnt = *(int32_t*)p_input_cnt;
        return graph->CreateOperation<tim::vx::ops::Stack>(axis, input_cnt);
    }

   private:
    op::pack::signature signature;
};

class PadCreator final : public OpCreator {
   public:
    PadCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
               const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_PAD, inputs, outputs) {
        if ((inputs.size() != 2) || outputs.size() != 1) {
            LOGE("PadCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in_pad = inputs[1];
        uint32_t idx_out = outputs[0];

        auto p_pad = (int32_t*)tensorMap.at(idx_in_pad).data.data();
        uint32_t pad_length = tensorMap.at(idx_in_pad).data.size() / 4;
        std::vector<int32_t> pad(p_pad, p_pad + pad_length);

        auto pad_attr = tensorMap.at(idx_in_pad).attr;
        uint32_t rank = tensorMap.at(idx_in).shape.size();
        if (pad_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("PadCreator: Cannot support pad tensor as INPUT in Pad");
            supported_ = false;
        } else {
            std::vector<uint32_t> front_size, back_size;
            for (int i = 0; i < rank; ++i) {
                front_size.push_back(*(p_pad + i * 2));
                back_size.push_back(*(p_pad + i * 2 + 1));
            }
            if (rank ==  4 && (front_size[0] != 0 || back_size[0] != 0)) {
                LOGI("PadCreator: Cannot support padding on batch in PadV2");
                supported_ = false;
            }
        }

        std::get<0>(signature.field_tuple) = op::pad::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::pad::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::pad::Pad(pad);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
        auto p_pad = (uint32_t*)std::get<2>(signature.field_tuple).storage.data.data();
        std::vector<uint32_t> front_size, back_size;
        for (int i = 0; i < rank; ++i) {
            front_size.push_back(*(p_pad + i * 2));
            back_size.push_back(*(p_pad + i * 2 + 1));
        }
        // The dim value reverses along with the shape value.
        std::reverse(front_size.begin(), front_size.end());
        std::reverse(back_size.begin(), back_size.end());

        return graph->CreateOperation<tim::vx::ops::Pad>(front_size, back_size, 0);
    }

   private:
    op::pad::signature signature;
};

class PadV2Creator final : public OpCreator {
   public:
    PadV2Creator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                 const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_PAD_V2, inputs, outputs) {
        if ((inputs.size() != 3) || outputs.size() != 1) {
            LOGE("PadV2Creator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in_pad = inputs[1];
        uint32_t idx_const_val = inputs[2];
        uint32_t idx_out = outputs[0];

        auto in_dtype = tensorMap.at(idx_in).dtype;
        auto const_dtype = scalarMap.at(idx_const_val).dtype;
        if ((in_dtype == slang::type::data_type::kINT8 ||
             in_dtype == slang::type::data_type::kUINT8) &&
            const_dtype == slang::type::data_type::kINT32) {
            // In the golden of vts case, the int32 const value is not quantized
            LOGI("PadV2Creator: Cannot support INT8/UINT8 input with INT32 const value in PadV2");
            supported_ = false;
        }
        auto p_pad = (int32_t*)tensorMap.at(idx_in_pad).data.data();
        uint32_t pad_length = tensorMap.at(idx_in_pad).data.size() / 4;
        std::vector<int32_t> pad(p_pad, p_pad + pad_length);

        auto pad_attr = tensorMap.at(idx_in_pad).attr;
        uint32_t rank = tensorMap.at(idx_in).shape.size();
        if (pad_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("PadV2Creator: Cannot support pad tensor as INPUT in PadV2");
            supported_ = false;
        } else {
            std::vector<uint32_t> front_size, back_size;
            for (int i = 0; i < rank; ++i) {
                front_size.push_back(*(p_pad + i * 2));
                back_size.push_back(*(p_pad + i * 2 + 1));
            }
            if (front_size[0] != 0 || back_size[0] != 0) {
                LOGI("PadV2Creator: Cannot support padding on highest dimension in PadV2");
                supported_ = false;
            }
        }


        std::get<0>(signature.field_tuple) = op::pad_v2::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::pad_v2::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::pad_v2::Pad(pad);
        std::get<3>(signature.field_tuple) = op::pad_v2::Const_val(scalarMap.at(idx_const_val));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
        uint32_t* p_pad = (uint32_t*)std::get<2>(signature.field_tuple).storage.data.data();
        uint8_t* p_const_val = std::get<3>(signature.field_tuple).storage.data.data();

        std::vector<uint32_t> front_size, back_size;
        for (int i = 0; i < rank; ++i) {
            front_size.push_back(*(p_pad + i * 2));
            back_size.push_back(*(p_pad + i * 2 + 1));
        }
        // The dim value reverses along with the shape value.
        std::reverse(front_size.begin(), front_size.end());
        std::reverse(back_size.begin(), back_size.end());
        auto datatype = std::get<0>(signature.field_tuple).storage.dtype;
        switch (datatype) {
            case slang::type::data_type::kFP16:
                return graph->CreateOperation<tim::vx::ops::PadV2>(front_size, back_size,
                                                                   *(_Float16*)p_const_val);
            case slang::type::data_type::kFP32:
                return graph->CreateOperation<tim::vx::ops::PadV2>(front_size, back_size,
                                                                   *(float*)p_const_val);
            default:
                return graph->CreateOperation<tim::vx::ops::Pad>(front_size, back_size,
                                                                 *(int32_t*)p_const_val);
        }
    }

   private:
    op::pad_v2::signature signature;
};

class PowCreator final : public OpCreator {
   public:
    PowCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
               const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_POW, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("PowCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::eltwise::Input0(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::eltwise::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::eltwise::Output(tensorMap.at(idx_out));
        std::get<3>(signature.field_tuple) = op::eltwise::Activation(0);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Pow>();
    }

   private:
    op::eltwise::signature signature;
};

class PreluCreator final : public OpCreator {
   public:
    PreluCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                 const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_PRELU, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("PreluCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_alpha = inputs[1];
        uint32_t idx_out = outputs[0];

        auto in_shape = tensorMap.at(idx_in).shape;
        auto alpha_shape = tensorMap.at(idx_alpha).shape;
        if (in_shape.size() < alpha_shape.size()) {
            LOGI("PreluCreator: Cannot support alpha tensor longer than INPUT");
            supported_ = false;
        }
        auto alpha_attr = tensorMap.at(idx_alpha).attr;
        if (alpha_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("PreluCreator: Cannot support alpha tensor as INPUT");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::prelu::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::prelu::Alpha(tensorMap.at(idx_alpha));
        std::get<2>(signature.field_tuple) = op::prelu::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return false;}
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Prelu>(0);
    }

   private:
    op::prelu::signature signature;
};

class QuantizeCreator final : public OpCreator {
   public:
    QuantizeCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                    const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_QUANTIZE, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("QuantizeCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        auto q_type = tensorMap.at(idx_out).qtype;
        if (q_type == slang::type::quant_type::kSYMM_PCQ) {
            LOGI("QuantizeCreator: Cannot support Perchannel quantize");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::quantize::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::quantize::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::DataConvert>();
    }

   private:
    op::quantize::signature signature;
};

class ReduceAllCreator final : public OpCreator {
   public:
    ReduceAllCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                     const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_REDUCE_ALL, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 1) {
            LOGE("ReduceAllCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_axis = inputs[1];
        uint32_t idx_keepdims = inputs[2];
        uint32_t idx_out = outputs[0];

        auto axis_attr = tensorMap.at(idx_axis).attr;
        if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("ReduceAllCreator: Cannot support axis tensor as INPUT");
            supported_ = false;
        }
        auto rank = tensorMap.at(idx_in).shape.size();
        const uint8_t* data = tensorMap.at(idx_axis).data.data();
        auto length = tensorMap.at(idx_axis).data.size() / 4;
        std::set<int32_t> unique_axis;
        for (uint32_t i = 0; i < length; ++i) {
            int32_t axis = *((int32_t*)data + i);
            if (axis < 0) axis += rank;
            unique_axis.insert(axis);
        }
        if (unique_axis.size() == rank) {
            LOGI("ReduceAllCreator: Cannot support all dimensions need reduce");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::reduce_all_any::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::reduce_all_any::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::reduce_all_any::Axis(tensorMap.at(idx_axis));
        std::get<3>(signature.field_tuple) =
                op::reduce_all_any::KeepDims(scalarMap.at(idx_keepdims));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        std::set<int32_t> unique_axis;
        const uint32_t in_rank = std::get<0>(signature.field_tuple).storage.shape.size();
        const uint32_t out_rank = std::get<1>(signature.field_tuple).storage.shape.size();
        const void* p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        const uint32_t data_length = std::get<2>(signature.field_tuple).storage.data.size() / 4;

        for (int i = 0; i < data_length; i++) {
            int32_t axis_android = *((int32_t*)p_axis + i);
            unique_axis.insert(convertToVxAxis(axis_android, in_rank));
        }
        std::vector<int32_t> axis_vx(unique_axis.begin(), unique_axis.end());
        const uint8_t* p_keepdims = std::get<3>(signature.field_tuple).storage.data.data();
        bool keepdims = *(bool*)p_keepdims;
        if (in_rank == out_rank) {
            keepdims = true;
        }
        return graph->CreateOperation<tim::vx::ops::ReduceAll>(axis_vx, keepdims);
    }

   private:
    op::reduce_all_any::signature signature;
};

class ReduceAnyCreator final : public OpCreator {
   public:
    ReduceAnyCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                     const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_REDUCE_ANY, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 1) {
            LOGE("ReduceAnyCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_axis = inputs[1];
        uint32_t idx_keepdims = inputs[2];
        uint32_t idx_out = outputs[0];

        auto axis_attr = tensorMap.at(idx_axis).attr;
        if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("ReduceAnyCreator: Cannot support axis tensor as INPUT");
            supported_ = false;
        }
        auto rank = tensorMap.at(idx_in).shape.size();
        const uint8_t* data = tensorMap.at(idx_axis).data.data();
        auto length = tensorMap.at(idx_axis).data.size() / 4;
        std::set<int32_t> unique_axis;
        for (uint32_t i = 0; i < length; ++i) {
            int32_t axis = *((int32_t*)data + i);
            if (axis < 0) axis += rank;
            unique_axis.insert(axis);
        }
        if (unique_axis.size() == rank) {
            LOGI("ReduceAllCreator: Cannot support all dimensions need reduce");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::reduce_all_any::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::reduce_all_any::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::reduce_all_any::Axis(tensorMap.at(idx_axis));
        std::get<3>(signature.field_tuple) =
                op::reduce_all_any::KeepDims(scalarMap.at(idx_keepdims));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        std::set<int32_t> unique_axis;
        const uint32_t in_rank = std::get<0>(signature.field_tuple).storage.shape.size();
        const uint32_t out_rank = std::get<1>(signature.field_tuple).storage.shape.size();
        const void* p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        const uint32_t data_length = std::get<2>(signature.field_tuple).storage.data.size() / 4;

        for (int i = 0; i < data_length; i++) {
            int32_t axis_android = *((int32_t*)p_axis + i);
            unique_axis.insert(convertToVxAxis(axis_android, in_rank));
        }
        std::vector<int32_t> axis_vx(unique_axis.begin(), unique_axis.end());
        const uint8_t* p_keepdims = std::get<3>(signature.field_tuple).storage.data.data();
        bool keepdims = *(bool*)p_keepdims;
        if (in_rank == out_rank) {
            keepdims = true;
        }
        return graph->CreateOperation<tim::vx::ops::ReduceAny>(axis_vx, keepdims);
    }

   private:
    op::reduce_all_any::signature signature;
};

class ReduceMaxCreator final : public OpCreator {
   public:
    ReduceMaxCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                     const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_REDUCE_MAX, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 1) {
            LOGE("ReduceMaxCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_axis = inputs[1];
        uint32_t idx_keepdims = inputs[2];
        uint32_t idx_out = outputs[0];

        auto axis_attr = tensorMap.at(idx_axis).attr;
        if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("ReduceMaxCreator: Cannot support axis tensor as INPUT");
            supported_ = false;
        }
        auto rank = tensorMap.at(idx_in).shape.size();
        const uint8_t* data = tensorMap.at(idx_axis).data.data();
        auto length = tensorMap.at(idx_axis).data.size() / 4;
        std::set<int32_t> unique_axis;
        for (uint32_t i = 0; i < length; ++i) {
            int32_t axis = *((int32_t*)data + i);
            if (axis < 0) axis += rank;
            unique_axis.insert(axis);
        }
        if (unique_axis.size() == rank) {
            LOGI("ReduceAllCreator: Cannot support all dimensions need reduce");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::Axis(tensorMap.at(idx_axis));
        std::get<3>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::KeepDims(scalarMap.at(idx_keepdims));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        std::vector<int32_t> axis_vx;
        const uint32_t in_rank = std::get<0>(signature.field_tuple).storage.shape.size();
        const uint32_t out_rank = std::get<1>(signature.field_tuple).storage.shape.size();
        const void* p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        const uint32_t data_length = std::get<2>(signature.field_tuple).storage.data.size() / 4;

        for (int i = 0; i < data_length; i++) {
            int32_t axis_android = *((int32_t*)p_axis + i);
            axis_vx.push_back(convertToVxAxis(axis_android, in_rank));
        }

        const uint8_t* p_keepdims = std::get<3>(signature.field_tuple).storage.data.data();
        bool keepdims = *(bool*)p_keepdims;
        if (in_rank == out_rank) {
            keepdims = true;
        }
        return graph->CreateOperation<tim::vx::ops::ReduceMax>(axis_vx, keepdims);
    }

   private:
    op::reduce_max_min_prod_sum::signature signature;
};

class ReduceMinCreator final : public OpCreator {
   public:
    ReduceMinCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                     const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_REDUCE_MIN, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 1) {
            LOGE("ReduceMinCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_axis = inputs[1];
        uint32_t idx_keepdims = inputs[2];
        uint32_t idx_out = outputs[0];

        auto axis_attr = tensorMap.at(idx_axis).attr;
        if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("ReduceMinCreator: Cannot support axis tensor as INPUT");
            supported_ = false;
        }
        auto rank = tensorMap.at(idx_in).shape.size();
        const uint8_t* data = tensorMap.at(idx_axis).data.data();
        auto length = tensorMap.at(idx_axis).data.size() / 4;
        std::set<int32_t> unique_axis;
        for (uint32_t i = 0; i < length; ++i) {
            int32_t axis = *((int32_t*)data + i);
            if (axis < 0) axis += rank;
            unique_axis.insert(axis);
        }
        if (unique_axis.size() == rank) {
            LOGI("ReduceAllCreator: Cannot support all dimensions need reduce");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::Axis(tensorMap.at(idx_axis));
        std::get<3>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::KeepDims(scalarMap.at(idx_keepdims));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        std::vector<int32_t> axis_vx;
        const uint32_t in_rank = std::get<0>(signature.field_tuple).storage.shape.size();
        const uint32_t out_rank = std::get<1>(signature.field_tuple).storage.shape.size();
        const void* p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        const uint32_t data_length = std::get<2>(signature.field_tuple).storage.data.size() / 4;

        for (int i = 0; i < data_length; i++) {
            int32_t axis_android = *((int32_t*)p_axis + i);
            axis_vx.push_back(convertToVxAxis(axis_android, in_rank));
        }

        const uint8_t* p_keepdims = std::get<3>(signature.field_tuple).storage.data.data();
        bool keepdims = *(bool*)p_keepdims;
        if (in_rank == out_rank) {
            keepdims = true;
        }
        return graph->CreateOperation<tim::vx::ops::ReduceMin>(axis_vx, keepdims);
    }

   private:
    op::reduce_max_min_prod_sum::signature signature;
};

class ReduceProdCreator final : public OpCreator {
   public:
    ReduceProdCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                      const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_REDUCE_PROD, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 1) {
            LOGE("ReduceProdCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_axis = inputs[1];
        uint32_t idx_keepdims = inputs[2];
        uint32_t idx_out = outputs[0];

        auto axis_attr = tensorMap.at(idx_axis).attr;
        if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("ReduceProdCreator: Cannot support axis tensor as INPUT");
            supported_ = false;
        }
        auto rank = tensorMap.at(idx_in).shape.size();
        const uint8_t* data = tensorMap.at(idx_axis).data.data();
        auto length = tensorMap.at(idx_axis).data.size() / 4;
        std::set<int32_t> unique_axis;
        for (uint32_t i = 0; i < length; ++i) {
            int32_t axis = *((int32_t*)data + i);
            if (axis < 0) axis += rank;
            unique_axis.insert(axis);
        }
        if (unique_axis.size() == rank) {
            LOGI("ReduceAllCreator: Cannot support all dimensions need reduce");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::Axis(tensorMap.at(idx_axis));
        std::get<3>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::KeepDims(scalarMap.at(idx_keepdims));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        std::vector<int32_t> axis_vx;
        const uint32_t in_rank = std::get<0>(signature.field_tuple).storage.shape.size();
        const uint32_t out_rank = std::get<1>(signature.field_tuple).storage.shape.size();
        const void* p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        const uint32_t data_length = std::get<2>(signature.field_tuple).storage.data.size() / 4;

        for (int i = 0; i < data_length; i++) {
            int32_t axis_android = *((int32_t*)p_axis + i);
            axis_vx.push_back(convertToVxAxis(axis_android, in_rank));
        }

        const uint8_t* p_keepdims = std::get<3>(signature.field_tuple).storage.data.data();
        bool keepdims = *(bool*)p_keepdims;
        if (in_rank == out_rank) {
            keepdims = true;
        }
        return graph->CreateOperation<tim::vx::ops::ReduceProd>(axis_vx, keepdims);
    }

   private:
    op::reduce_max_min_prod_sum::signature signature;
};

class ReduceSumCreator final : public OpCreator {
   public:
    ReduceSumCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                     const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_REDUCE_SUM, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 1) {
            LOGE("ReduceSumCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_axis = inputs[1];
        uint32_t idx_keepdims = inputs[2];
        uint32_t idx_out = outputs[0];

        auto axis_attr = tensorMap.at(idx_axis).attr;
        if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("ReduceSumCreator: Cannot support axis tensor as INPUT");
            supported_ = false;
        }
        auto rank = tensorMap.at(idx_in).shape.size();
        const uint8_t* data = tensorMap.at(idx_axis).data.data();
        auto length = tensorMap.at(idx_axis).data.size() / 4;
        std::set<int32_t> unique_axis;
        for (uint32_t i = 0; i < length; ++i) {
            int32_t axis = *((int32_t*)data + i);
            if (axis < 0) axis += rank;
            unique_axis.insert(axis);
        }
        if (unique_axis.size() == rank) {
            LOGI("ReduceAllCreator: Cannot support all dimensions need reduce");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::Axis(tensorMap.at(idx_axis));
        std::get<3>(signature.field_tuple) =
                op::reduce_max_min_prod_sum::KeepDims(scalarMap.at(idx_keepdims));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        std::vector<int32_t> axis_vx;
        const uint32_t in_rank = std::get<0>(signature.field_tuple).storage.shape.size();
        const uint32_t out_rank = std::get<1>(signature.field_tuple).storage.shape.size();
        const void* p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        const uint32_t data_length = std::get<2>(signature.field_tuple).storage.data.size() / 4;

        for (int i = 0; i < data_length; i++) {
            int32_t axis_android = *((int32_t*)p_axis + i);
            axis_vx.push_back(convertToVxAxis(axis_android, in_rank));
        }

        const uint8_t* p_keepdims = std::get<3>(signature.field_tuple).storage.data.data();
        bool keepdims = *(bool*)p_keepdims;
        if (in_rank == out_rank) {
            keepdims = true;
        }
        return graph->CreateOperation<tim::vx::ops::ReduceSum>(axis_vx, keepdims);
    }

   private:
    op::reduce_max_min_prod_sum::signature signature;
};

class ReluCreator final : public OpCreator {
   public:
    ReluCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_RELU, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("ReluCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::activation::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::activation::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Relu>();
    }

   private:
    op::activation::signature signature;
};

class Relu1Creator final : public OpCreator {
   public:
    Relu1Creator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                 const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_RELU1, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("Relu1Creator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::activation::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::activation::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Relu1>();
    }

   private:
    op::activation::signature signature;
};

class Relu6Creator final : public OpCreator {
   public:
    Relu6Creator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                 const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_RELU6, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("Relu6Creator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::activation::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::activation::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Relu6>();
    }

   private:
    op::activation::signature signature;
};

class ReshapeCreator final : public OpCreator {
   public:
    ReshapeCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                   const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_RESHAPE, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("ReshapeCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_shape = inputs[1];
        uint32_t idx_out = outputs[0];

        auto shape_attr = tensorMap.at(idx_shape).attr;
        if (shape_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("ReshapeCreator: Cannot support shape tensor as INPUT");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::reshape::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::reshape::Shape(tensorMap.at(idx_shape));
        std::get<2>(signature.field_tuple) = op::reshape::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        auto shape_tensor = std::get<1>(signature.field_tuple);
        const void* data = shape_tensor.data();
        uint32_t length = shape_tensor.data_length() / 4;  // The type of shape tensor is int32

        std::vector<int32_t> shape((int32_t*)data, (int32_t*)data + length);
        std::vector<uint32_t> no_negative_shape;
        uint32_t total_size = 1, negative_index = 0;
        bool do_shape_inference = false;
        auto input_shape = std::get<0>(signature.field_tuple).shape();

        for (uint32_t i = 0; i < input_shape.size(); ++i) {
            total_size *= input_shape.data()[i];
        }
        for (uint32_t i = 0; i < shape.size(); ++i) {
            if (shape.at(i) != -1) {
                total_size /= shape.at(i);
                no_negative_shape.push_back(shape.at(i));
            } else {
                do_shape_inference = true;
                negative_index = i;
                no_negative_shape.push_back(0);  // hold a place for value changes
            }
        }
        if (do_shape_inference) {
            no_negative_shape.at(negative_index) = total_size;
        }
        std::reverse(no_negative_shape.begin(), no_negative_shape.end());
        return graph->CreateOperation<tim::vx::ops::Reshape>(no_negative_shape);
    }

   private:
    op::reshape::signature signature;
};

class ResizeBilinearCreator final : public OpCreator {
   public:
    ResizeBilinearCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                          const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_RESIZE_BILINEAR, inputs, outputs) {
        if (inputs.size() < 3 || inputs.size() > 6 || outputs.size() != 1) {
            LOGE("ResizeBilinearCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_output_width = inputs[1];
        uint32_t idx_output_height = inputs[2];
        uint32_t idx_out = outputs[0];

        bool layout = false;
        bool align_corners = false;
        bool half_pixel_centers = false;
        int32_t output_width = 0, output_height = 0;
        float factor_width = 0, factor_height = 0;
        if (scalarMap.at(inputs[1]).dtype == slang::type::data_type::kINT32) {
            std::get<4>(signature.field_tuple) = op::resize::Factor(0.0f);
            auto p_output_width = scalarMap.at(inputs[1]).data.data();
            auto p_output_height = scalarMap.at(inputs[2]).data.data();
            output_width = *(int32_t*)p_output_width;
            output_height = *(int32_t*)p_output_height;
        } else {
            std::get<4>(signature.field_tuple) = op::resize::Factor(scalarMap.at(inputs[1]));
            auto p_factor_width = scalarMap.at(inputs[1]).data.data();
            auto p_factor_height = scalarMap.at(inputs[2]).data.data();
            if (scalarMap.at(inputs[1]).dtype == slang::type::data_type::kFP16) {
                factor_width = *(_Float16*)p_factor_width;
                factor_height = *(_Float16*)p_factor_height;
            } else {
                factor_width = *(float*)p_factor_width;
                factor_height = *(float*)p_factor_height;
            }
            if (abs(factor_width - factor_height) > 1e-5f) {
                LOGI("ResizeBilinearCreator: cannot support factor_width not equal to "
                     "factor_height");
                supported_ = false;
            }
        }
        if (inputs.size() > 3) {
            uint32_t idx_layout = inputs[3];
            auto p_layout = scalarMap.at(idx_layout).data.data();
            layout = *(bool*)p_layout;
        }
        if (inputs.size() > 4) {
            uint32_t idx_align_corners = inputs[4];
            auto p_align_corners = scalarMap.at(idx_align_corners).data.data();
            align_corners = *(bool*)p_align_corners;
        }
        if (inputs.size() > 5) {
            uint32_t idx_half_pixel_centers = inputs[5];
            auto p_half_pixel_centers = scalarMap.at(idx_half_pixel_centers).data.data();
            half_pixel_centers = *(bool*)p_half_pixel_centers;
        }
        std::get<0>(signature.field_tuple) = op::resize::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::resize::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::resize::Output_width(output_width);
        std::get<3>(signature.field_tuple) = op::resize::Output_height(output_height);
        std::get<5>(signature.field_tuple) = op::resize::Layout(layout);
        std::get<6>(signature.field_tuple) = op::resize::Align_corners(align_corners);
        std::get<7>(signature.field_tuple) = op::resize::Half_pixel_centers(half_pixel_centers);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        uint8_t* p_output_width = std::get<2>(signature.field_tuple).storage.data.data();
        uint8_t* p_output_height = std::get<3>(signature.field_tuple).storage.data.data();
        uint8_t* p_factor = std::get<4>(signature.field_tuple).storage.data.data();
        uint8_t* p_layout = std::get<5>(signature.field_tuple).storage.data.data();
        uint8_t* p_align_corners = std::get<6>(signature.field_tuple).storage.data.data();
        uint8_t* p_half_pixel_centers = std::get<7>(signature.field_tuple).storage.data.data();
        int32_t output_width = *(int32_t*)p_output_width;
        int32_t output_height = *(int32_t*)p_output_height;
        bool align_corners = *(bool*)p_align_corners;
        bool half_pixel_centers = *(bool*)p_half_pixel_centers;
        auto layout = convertToVxLayout(*(bool*)p_layout);
        auto input_dtype = std::get<0>(signature.field_tuple).storage.dtype;
        if (input_dtype == slang::type::data_type::kFP16) {
            return graph->CreateOperation<tim::vx::ops::Resize>(
                    tim::vx::ResizeType::BILINEAR, *(_Float16*)p_factor, align_corners,
                    half_pixel_centers, output_height, output_width, layout);
        } else {
            return graph->CreateOperation<tim::vx::ops::Resize>(
                    tim::vx::ResizeType::BILINEAR, *(float*)p_factor, align_corners,
                    half_pixel_centers, output_height, output_width, layout);
        }
    }

   private:
    op::resize::signature signature;
};

class ResizeNearestCreator final : public OpCreator {
   public:
    ResizeNearestCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                         const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR, inputs, outputs) {
        if (inputs.size() < 3 || inputs.size() > 6 || outputs.size() != 1) {
            LOGE("ResizeNearestCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_output_width = inputs[1];
        uint32_t idx_output_height = inputs[2];
        uint32_t idx_out = outputs[0];

        bool layout = false;
        bool align_corners = false;
        bool half_pixel_centers = false;
        int32_t output_width = 0, output_height = 0;
        float factor_width = 0, factor_height = 0;
        if (scalarMap.at(inputs[1]).dtype == slang::type::data_type::kINT32) {
            std::get<4>(signature.field_tuple) = op::resize::Factor(0.0f);
            auto* p_output_width = scalarMap.at(inputs[1]).data.data();
            auto* p_output_height = scalarMap.at(inputs[2]).data.data();
            output_width = *(int32_t*)p_output_width;
            output_height = *(int32_t*)p_output_height;
        } else {
            std::get<4>(signature.field_tuple) = op::resize::Factor(scalarMap.at(inputs[1]));
            auto* p_factor_width = scalarMap.at(inputs[1]).data.data();
            auto* p_factor_height = scalarMap.at(inputs[2]).data.data();
            if (scalarMap.at(inputs[1]).dtype == slang::type::data_type::kFP16) {
                factor_width = *(_Float16*)p_factor_width;
                factor_height = *(_Float16*)p_factor_height;
            } else {
                factor_width = *(float*)p_factor_width;
                factor_height = *(float*)p_factor_height;
            }
            if (abs(factor_width - factor_height) > 1e-5f) {
                LOGI("ResizeNearestCreator: cannot support factor_width not equal to "
                     "factor_height");
                supported_ = false;
            }
        }
        if (inputs.size() > 3) {
            uint32_t idx_layout = inputs[3];
            auto* p_layout = scalarMap.at(idx_layout).data.data();
            layout = *(bool*)p_layout;
        }
        if (inputs.size() > 4) {
            uint32_t idx_align_corners = inputs[4];
            auto p_align_corners = scalarMap.at(idx_align_corners).data.data();
            align_corners = *(bool*)p_align_corners;
        }
        if (inputs.size() > 5) {
            uint32_t idx_half_pixel_centers = inputs[5];
            auto p_half_pixel_centers = scalarMap.at(idx_half_pixel_centers).data.data();
            half_pixel_centers = *(bool*)p_half_pixel_centers;
        }
        std::get<0>(signature.field_tuple) = op::resize::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::resize::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::resize::Output_width(output_width);
        std::get<3>(signature.field_tuple) = op::resize::Output_height(output_height);
        std::get<5>(signature.field_tuple) = op::resize::Layout(layout);
        std::get<6>(signature.field_tuple) = op::resize::Align_corners(align_corners);
        std::get<7>(signature.field_tuple) = op::resize::Half_pixel_centers(half_pixel_centers);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        uint8_t* p_output_width = std::get<2>(signature.field_tuple).storage.data.data();
        uint8_t* p_output_height = std::get<3>(signature.field_tuple).storage.data.data();
        uint8_t* p_factor = std::get<4>(signature.field_tuple).storage.data.data();
        uint8_t* p_layout = std::get<5>(signature.field_tuple).storage.data.data();
        uint8_t* p_align_corners = std::get<6>(signature.field_tuple).storage.data.data();
        uint8_t* p_half_pixel_centers = std::get<7>(signature.field_tuple).storage.data.data();
        int32_t output_width = *(int32_t*)p_output_width;
        int32_t output_height = *(int32_t*)p_output_height;
        bool align_corners = *(bool*)p_align_corners;
        bool half_pixel_centers = *(bool*)p_half_pixel_centers;
        auto layout = convertToVxLayout(*(bool*)p_layout);
        auto input_dtype = std::get<0>(signature.field_tuple).storage.dtype;

        if (input_dtype == slang::type::data_type::kFP16) {
            return graph->CreateOperation<tim::vx::ops::Resize>(
                    tim::vx::ResizeType::NEAREST_NEIGHBOR, *(_Float16*)p_factor, align_corners,
                    half_pixel_centers, output_height, output_width, layout);
        } else {
            return graph->CreateOperation<tim::vx::ops::Resize>(
                    tim::vx::ResizeType::NEAREST_NEIGHBOR, *(float*)p_factor, align_corners,
                    half_pixel_centers, output_height, output_width, layout);
        }
    }

   private:
    op::resize::signature signature;
};

class ReverseCreator final : public OpCreator {
   public:
    ReverseCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                   const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_REVERSE, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("ReverseCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_axis = inputs[1];
        uint32_t idx_out = outputs[0];

        auto axis_attr = tensorMap.at(idx_axis).attr;
        if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("ReverseCreator: Cannot support axis tensor as INPUT");
            supported_ = false;
        }
        auto p_axis = tensorMap.at(idx_axis).data.data();
        auto axis_android = *(int32_t*)p_axis;
        int32_t axis_vx = convertToVxAxis(axis_android, tensorMap.at(idx_in).shape.size());

        std::get<0>(signature.field_tuple) = op::reverse::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::reverse::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::reverse::Axis(axis_vx);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        auto p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        int32_t axis_vx = *(int32_t*)p_axis;
        std::vector<int> axis{axis_vx};
        return graph->CreateOperation<tim::vx::ops::Reverse>(axis);
    }

   private:
    op::reverse::signature signature;
};

class RoiAlignCreator final : public OpCreator {
   public:
    RoiAlignCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                    const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_ROI_ALIGN, inputs, outputs) {
        if (inputs.size() != 10 || outputs.size() != 1) {
            LOGE("RoiAlignCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_regions = inputs[1];
        uint32_t idx_batch_index = inputs[2];
        uint32_t idx_out_h = inputs[3];
        uint32_t idx_out_w = inputs[4];
        uint32_t idx_h_ratio = inputs[5];
        uint32_t idx_w_ratio = inputs[6];
        uint32_t idx_h_sample = inputs[7];
        uint32_t idx_w_sample = inputs[8];
        uint32_t idx_layout = inputs[9];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::roi_align::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::roi_align::Regions(tensorMap.at(idx_regions));
        std::get<2>(signature.field_tuple) =
                op::roi_align::BatchIndex(tensorMap.at(idx_batch_index));
        std::get<3>(signature.field_tuple) = op::roi_align::Output(tensorMap.at(idx_out));
        std::get<4>(signature.field_tuple) = op::roi_align::OutputHeight(scalarMap.at(idx_out_h));
        std::get<5>(signature.field_tuple) = op::roi_align::OutputWidth(scalarMap.at(idx_out_w));
        std::get<6>(signature.field_tuple) = op::roi_align::HeightRatio(scalarMap.at(idx_h_ratio));
        std::get<7>(signature.field_tuple) = op::roi_align::WidthRatio(scalarMap.at(idx_w_ratio));
        std::get<8>(signature.field_tuple) = op::roi_align::HSampleNum(scalarMap.at(idx_h_sample));
        std::get<9>(signature.field_tuple) = op::roi_align::WSampleNum(scalarMap.at(idx_w_sample));
        std::get<10>(signature.field_tuple) = op::roi_align::Layout(scalarMap.at(idx_layout));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        uint8_t* p_out_height = std::get<4>(signature.field_tuple).storage.data.data();
        uint8_t* p_out_width = std::get<5>(signature.field_tuple).storage.data.data();
        uint8_t* p_height_ratio = std::get<6>(signature.field_tuple).storage.data.data();
        uint8_t* p_width_ratio = std::get<7>(signature.field_tuple).storage.data.data();
        uint8_t* p_height_sample_num = std::get<8>(signature.field_tuple).storage.data.data();
        uint8_t* p_width_sample_num = std::get<9>(signature.field_tuple).storage.data.data();
        uint8_t* p_layout = std::get<10>(signature.field_tuple).storage.data.data();
        int32_t out_h = *(int32_t*)p_out_height;
        int32_t out_w = *(int32_t*)p_out_width;
        int32_t h_sample_num = *(int32_t*)p_height_sample_num;
        int32_t w_sample_num = *(int32_t*)p_width_sample_num;
        auto layout = convertToVxLayout(*(bool*)p_layout);
        auto datatype = std::get<0>(signature.field_tuple).storage.dtype;
        if (datatype == slang::type::data_type::kFP16) {
            auto h_ratio = *(_Float16*)p_height_ratio;
            auto w_ratio = *(_Float16*)p_width_ratio;
            return graph->CreateOperation<tim::vx::ops::RoiAlign>(
                    out_h, out_w, h_ratio, w_ratio, h_sample_num, w_sample_num, layout);
        } else {
            auto h_ratio = *(float*)p_height_ratio;
            auto w_ratio = *(float*)p_width_ratio;
            return graph->CreateOperation<tim::vx::ops::RoiAlign>(
                    out_h, out_w, h_ratio, w_ratio, h_sample_num, w_sample_num, layout);
        }
    }

   private:
    op::roi_align::signature signature;
};

class RoiPoolingCreator final : public OpCreator {
   public:
    RoiPoolingCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                      const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_ROI_POOLING, inputs, outputs) {
        if (inputs.size() != 8 || outputs.size() != 1) {
            LOGE("RoiPoolingCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_regions = inputs[1];
        uint32_t idx_batch_index = inputs[2];
        uint32_t idx_out_h = inputs[3];
        uint32_t idx_out_w = inputs[4];
        uint32_t idx_h_ratio = inputs[5];
        uint32_t idx_w_ratio = inputs[6];
        uint32_t idx_layout = inputs[7];
        uint32_t idx_out = outputs[0];
        auto p_h_ratio = scalarMap.at(idx_h_ratio).data.data();
        auto p_w_ratio = scalarMap.at(idx_w_ratio).data.data();
        auto input_type = tensorMap.at(idx_in).dtype;
        float h_ratio, w_ratio;
        if (input_type == slang::type::data_type::kFP16) {
            h_ratio = *(_Float16*)p_h_ratio;
            w_ratio = *(_Float16*)p_w_ratio;
            if (h_ratio != w_ratio) {
                LOGI("RoiPoolingCreator: Cannot support h_ratio & w_ratio not equal");
                supported_ = false;
            }
        } else {
            h_ratio = *(float*)p_h_ratio;
            w_ratio = *(float*)p_w_ratio;
            if (h_ratio != w_ratio) {
                LOGI("RoiPoolingCreator: Cannot support h_ratio & w_ratio not equal");
                supported_ = false;
            }
        }
        auto attr = tensorMap.at(idx_batch_index).attr;
        if (attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("RoiPoolingCreator: Cannot support batch_index as INPUT");
            supported_ = false;
        } else {
            const auto* data = tensorMap.at(idx_batch_index).data.data();
            auto length = tensorMap.at(idx_batch_index).data.size() / 4;
            for (int i = 0; i < length; ++i) {
                if (*((int32_t*)data + i) != 0) {
                    LOGI("RoiPoolingCreator: Cannot support batch_index not equal to zero");
                    supported_ = false;
                }
            }
        }
        std::get<0>(signature.field_tuple) = op::roi_pooling::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::roi_pooling::Regions(tensorMap.at(idx_regions));
        std::get<2>(signature.field_tuple) =
                op::roi_pooling::BatchIndex(tensorMap.at(idx_batch_index));
        std::get<3>(signature.field_tuple) = op::roi_pooling::Output(tensorMap.at(idx_out));
        std::get<4>(signature.field_tuple) = op::roi_pooling::OutputHeight(scalarMap.at(idx_out_h));
        std::get<5>(signature.field_tuple) = op::roi_pooling::OutputWidth(scalarMap.at(idx_out_w));
        std::get<6>(signature.field_tuple) = op::roi_pooling::Scale(scalarMap.at(idx_h_ratio));
        std::get<7>(signature.field_tuple) = op::roi_pooling::Layout(scalarMap.at(idx_layout));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        uint8_t* p_out_height = std::get<4>(signature.field_tuple).storage.data.data();
        uint8_t* p_out_width = std::get<5>(signature.field_tuple).storage.data.data();
        uint8_t* p_scale = std::get<6>(signature.field_tuple).storage.data.data();
        uint8_t* p_layout = std::get<7>(signature.field_tuple).storage.data.data();
        auto out_w = *(uint32_t*)p_out_width;
        auto out_h = *(uint32_t*)p_out_height;
        std::array<uint32_t, 2> size{out_w, out_h};
        auto layout = convertToVxLayout(*(bool*)p_layout);
        auto datatype = std::get<0>(signature.field_tuple).storage.dtype;
        if (datatype == slang::type::data_type::kFP16) {
            return graph->CreateOperation<tim::vx::ops::RoiPool>(tim::vx::PoolType::MAX,
                                                                 *(_Float16*)p_scale, size, layout);
        } else {
            return graph->CreateOperation<tim::vx::ops::RoiPool>(tim::vx::PoolType::MAX,
                                                                 *(float*)p_scale, size, layout);
        }
    }

   private:
    op::roi_pooling::signature signature;
};

class RsqrtCreator final : public OpCreator {
   public:
    RsqrtCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                 const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_RSQRT, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("RsqrtCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::simple_op::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::simple_op::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Rsqrt>();
    }

   private:
    op::simple_op::signature signature;
};

class SelectCreator final : public OpCreator {
   public:
    SelectCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                  const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_SELECT, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 1) {
            LOGE("SelectCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_choose = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_in2 = inputs[2];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::select::Choose(tensorMap.at(idx_choose));
        std::get<1>(signature.field_tuple) = op::select::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::select::Input2(tensorMap.at(idx_in2));
        std::get<3>(signature.field_tuple) = op::select::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Select>();
    }

   private:
    op::select::signature signature;
};

class SinCreator final : public OpCreator {
   public:
    SinCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
               const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_SIN, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("SinCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::simple_op::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::simple_op::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Sin>();
    }

   private:
    op::simple_op::signature signature;
};

class SliceCreator final : public OpCreator {
   public:
    SliceCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                 const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_SLICE, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 1) {
            LOGE("SliceCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_begin = inputs[1];
        uint32_t idx_size = inputs[2];
        uint32_t idx_out = outputs[0];
        auto begin_attr = tensorMap.at(idx_begin).attr;
        if (begin_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("SliceCreator: Cannot support begin tensor as INPUT");
            supported_ = false;
        }
        auto size_attr = tensorMap.at(idx_size).attr;
        if (size_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("SliceCreator: Cannot support size tensor as INPUT");
            supported_ = false;
        }
        std::get<0>(signature.field_tuple) = op::slice::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::slice::Begin(tensorMap.at(idx_begin));
        std::get<2>(signature.field_tuple) = op::slice::Size(tensorMap.at(idx_size));
        std::get<3>(signature.field_tuple) = op::slice::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const auto* p_begin = std::get<1>(signature.field_tuple).storage.data.data();
        const auto* p_size = std::get<2>(signature.field_tuple).storage.data.data();
        auto begin_length = std::get<1>(signature.field_tuple).storage.data.size() / 4;
        auto size_length = std::get<2>(signature.field_tuple).storage.data.size() / 4;
        std::vector<int32_t> begin((int32_t*)p_begin, (int32_t*)p_begin + begin_length);
        std::vector<int32_t> size((int32_t*)p_size, (int32_t*)p_size + size_length);
        auto input_shape = std::get<0>(signature.field_tuple).storage.shape;
        for (int i = 0; i < size.size(); ++i) {
            if (size[i] < 0) {
                size[i] = input_shape[i] - begin[i];
            }
        }  // size may be negative
        std::reverse(begin.begin(), begin.end());
        std::reverse(size.begin(), size.end());
        return graph->CreateOperation<tim::vx::ops::Slice>(input_shape.size(), begin, size);
    }

   private:
    op::slice::signature signature;
};

class SpaceToDepthCreator final : public OpCreator {
   public:
    SpaceToDepthCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                        const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_SPACE_TO_DEPTH, inputs, outputs) {
        if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1) {
            LOGE("SpaceToDepthCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_block_size = inputs[1];
        uint32_t idx_layout;
        uint32_t idx_out = outputs[0];

        bool layout = false;
        if (inputs.size() == 3) {
            idx_layout = inputs[2];
            const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
            layout = *(bool*)p_layout;
        }
        std::get<0>(signature.field_tuple) = op::space_to_depth::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::space_to_depth::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) =
                op::space_to_depth::BlockSize(scalarMap.at(idx_block_size));
        std::get<3>(signature.field_tuple) = op::space_to_depth::Layout(layout);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_block_size = std::get<2>(signature.field_tuple).storage.data.data();
        const uint8_t* p_layout = std::get<3>(signature.field_tuple).storage.data.data();
        std::vector<int32_t> block_size = {*(int32_t*)p_block_size, *(int32_t*)p_block_size};
        auto layout = convertToVxLayout(*(bool*)p_layout);
        return graph->CreateOperation<tim::vx::ops::SpaceToDepth>(block_size, layout);
    }

   private:
    op::space_to_depth::signature signature;
};

class SpaceToBatchCreator final : public OpCreator {
   public:
    SpaceToBatchCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                        const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_SPACE_TO_BATCH_ND, inputs, outputs) {
        if ((inputs.size() != 3 && inputs.size() != 4) || outputs.size() != 1) {
            LOGE("SpaceToBatchCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_block_size = inputs[1];
        uint32_t idx_pad = inputs[2];
        uint32_t idx_layout;
        uint32_t idx_out = outputs[0];

        auto block_size_attr = tensorMap.at(idx_block_size).attr;
        auto pad_attr = tensorMap.at(idx_pad).attr;
        if (block_size_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("SpaceToBatchCreator: Cannot support block tensor as INPUT");
            supported_ = false;
        }
        if (pad_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("SpaceToBatchCreator: Cannot support pad tensor as INPUT");
            supported_ = false;
        }
        const void* p_block_size = tensorMap.at(idx_block_size).data.data();
        const uint32_t block_size_length = tensorMap.at(idx_block_size).data.size() / 4;
        std::vector<int32_t> block_size((int32_t*)p_block_size,
                                        (int32_t*)p_block_size + block_size_length);

        const void* p_pad = tensorMap.at(idx_pad).data.data();
        const uint32_t pad_length = tensorMap.at(idx_pad).data.size() / 4;
        std::vector<int32_t> pad((int32_t*)p_pad, (int32_t*)p_pad + pad_length);
        bool layout = false;
        if (inputs.size() == 4) {
            idx_layout = inputs[3];
            const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
            layout = *(bool*)p_layout;
        }
        std::get<0>(signature.field_tuple) = op::space_to_batch::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::space_to_batch::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::space_to_batch::BlockSize(block_size);
        std::get<3>(signature.field_tuple) = op::space_to_batch::Pad(pad);
        std::get<4>(signature.field_tuple) = op::space_to_batch::Layout(layout);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_block_size = std::get<2>(signature.field_tuple).storage.data.data();
        const uint8_t* p_pad = std::get<3>(signature.field_tuple).storage.data.data();
        const uint32_t pad_length = std::get<3>(signature.field_tuple).storage.data.size() / 4;
        const uint8_t* p_layout = std::get<4>(signature.field_tuple).storage.data.data();
        // block_size reverse as input shape reverse
        std::vector<int32_t> block_size = {*((int32_t*)p_block_size + 1), *(int32_t*)p_block_size};
        std::vector<int32_t> pad((int32_t*)p_pad, (int32_t*)p_pad + pad_length);
        // Vts pad as HW, timvx pad as WH
        std::vector<int32_t> vx_pad = {pad[2], pad[3], pad[0], pad[1]};
        auto layout = convertToVxLayout(*(bool*)p_layout);
        return graph->CreateOperation<tim::vx::ops::Space2Batch>(block_size, vx_pad, layout);
    }

   private:
    op::space_to_batch::signature signature;
};

class SplitCreator final : public OpCreator {
   public:
    SplitCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                 const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_SPLIT, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() == 0) {
            LOGE("SplitCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_axis = inputs[1];
        uint32_t idx_num_splits = inputs[2];
        uint32_t idx_out = outputs[0];
        const uint32_t rank = tensorMap.at(idx_in).shape.size();
        auto p_axis = scalarMap.at(idx_axis).data.data();
        const uint8_t* p_num_splits = scalarMap.at(idx_num_splits).data.data();
        int32_t axis = *(int32_t*)p_axis;
        int32_t num_splits = *(int32_t*)p_num_splits;
        int32_t axis_vx = convertToVxAxis(axis, rank);

        auto& input_shape = tensorMap.at(idx_in).shape;
        axis = axis < 0 ? axis + rank : axis;
        int32_t dim_value = input_shape[axis];
        if (dim_value % num_splits != 0) {
            LOGE("SplitCreator: The number of splits can not evenly divide axis size.");
            supported_ = false;
        }
        uint32_t slice_length = dim_value / num_splits;
        std::vector<uint32_t> slices(num_splits, slice_length);

        std::get<0>(signature.field_tuple) = op::split::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::split::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::split::Axis(axis_vx);
        std::get<3>(signature.field_tuple) = op::split::Slices(slices);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        auto p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        auto p_slices = std::get<3>(signature.field_tuple).storage.data.data();
        auto slices_length = std::get<3>(signature.field_tuple).storage.data.size() / 4;
        int32_t axis = *(int32_t*)p_axis;
        std::vector<uint32_t> slices((int32_t*)p_slices, (int32_t*)p_slices + slices_length);
        return graph->CreateOperation<tim::vx::ops::Split>(axis, slices);
    }

   private:
    op::split::signature signature;
};

class SqueezeCreator final : public OpCreator {
   public:
    SqueezeCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                   const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_SQUEEZE, inputs, outputs) {
        if ((inputs.size() != 1 && inputs.size() != 2) || outputs.size() != 1) {
            LOGE("SqueezeCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_axis;
        uint32_t idx_out = outputs[0];
        std::vector<int32_t> axis_android;
        auto input_shape = tensorMap.at(idx_in).shape;
        if (inputs.size() == 2 && tensorMap.at(inputs[1]).data.size() != 0) {
            idx_axis = inputs[1];
            auto axis_attr = tensorMap.at(idx_axis).attr;
            if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
                LOGI("SqueezeCreator: Cannot support axis tensor as INPUT");
                supported_ = false;
            }
            const void* p_axis = tensorMap.at(idx_axis).data.data();
            const uint32_t axis_length = tensorMap.at(idx_axis).data.size() / 4;
            axis_android.assign((int32_t*)p_axis, (int32_t*)p_axis + axis_length);
            for (int i = 0; i < axis_android.size(); ++i) {
                if (input_shape[axis_android[i]] != 1) {
                    LOGI("SqueezeCreator: Cannot support Squeezing a dimension that is not 1.");
                    supported_ = false;
                }
            }
        } else {
            for (int i = 0; i < input_shape.size(); ++i) {
                if (input_shape[i] == 1) {
                    axis_android.push_back(i);
                }
            }
        }
        std::get<0>(signature.field_tuple) = op::squeeze::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::squeeze::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::squeeze::Axis(axis_android);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_axis = std::get<2>(signature.field_tuple).storage.data.data();
        const uint32_t data_length = std::get<2>(signature.field_tuple).storage.data.size() / 4;
        const uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
        std::vector<uint32_t> axis_vx;
        for (int i = 0; i < data_length; i++) {
            int32_t axis_android = *((int32_t*)p_axis + i);
            axis_vx.push_back(convertToVxAxis(axis_android, rank));
        }
        return graph->CreateOperation<tim::vx::ops::Squeeze>(axis_vx);
    }

   private:
    op::squeeze::signature signature;
};

class SqrtCreator final : public OpCreator {
   public:
    SqrtCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_SQRT, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("SqrtCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];

        std::get<0>(signature.field_tuple) = op::simple_op::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::simple_op::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Sqrt>();
    }

   private:
    op::simple_op::signature signature;
};

class SoftmaxCreator final : public OpCreator {
   public:
    SoftmaxCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                   const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_SOFTMAX, inputs, outputs) {
        if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1) {
            LOGE("SoftmaxCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_beta = inputs[1];
        uint32_t idx_out = outputs[0];
        uint32_t idx_axis;

        std::get<0>(signature.field_tuple) = op::softmax::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::softmax::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::softmax::Beta(scalarMap.at(idx_beta));
        std::get<3>(signature.field_tuple) = op::softmax::Axis(-1);  // default is -1

        if (inputs.size() == 3) {
            idx_axis = inputs[2];
            std::get<3>(signature.field_tuple) = op::softmax::Axis(scalarMap.at(idx_axis));
        }
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
        auto datatype = std::get<2>(signature.field_tuple).storage.dtype;
        const uint8_t* p_beta = std::get<2>(signature.field_tuple).storage.data.data();
        const uint8_t* p_axis = std::get<3>(signature.field_tuple).storage.data.data();
        int32_t axis_android = *(int32_t*)p_axis;
        int32_t axis_vx = convertToVxAxis(axis_android, rank);
        if (datatype == slang::type::data_type::kFP16) {
            auto beta = *(_Float16*)p_beta;
            return graph->CreateOperation<tim::vx::ops::Softmax>(beta, axis_vx);
        } else {
            auto beta = *(float*)p_beta;
            return graph->CreateOperation<tim::vx::ops::Softmax>(beta, axis_vx);
        }
    }

   private:
    op::softmax::signature signature;
};

class StridedSliceCreator final : public OpCreator {
   public:
    StridedSliceCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                        const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_STRIDED_SLICE, inputs, outputs) {
        if (inputs.size() != 7 || outputs.size() != 1) {
            LOGE("StridedSliceCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_begin = inputs[1];
        uint32_t idx_end = inputs[2];
        uint32_t idx_strides = inputs[3];
        uint32_t idx_begin_mask = inputs[4];
        uint32_t idx_end_mask = inputs[5];
        uint32_t idx_shrink_mask = inputs[6];
        uint32_t idx_out = outputs[0];

        auto attr_begin = tensorMap.at(idx_begin).attr;
        auto attr_end = tensorMap.at(idx_end).attr;
        auto attr_strides = tensorMap.at(idx_strides).attr;
        if (attr_begin != slang::type::tensor_attr::kCONSTANT) {
            LOGI("StridedSliceCreator: Cannot support begin tensor as INPUT");
            supported_ = false;
        }
        if (attr_end != slang::type::tensor_attr::kCONSTANT) {
            LOGI("StridedSliceCreator: Cannot support end tensor as INPUT");
            supported_ = false;
        }
        if (attr_strides != slang::type::tensor_attr::kCONSTANT) {
            LOGI("StridedSliceCreator: Cannot support strides tensor as INPUT");
            supported_ = false;
        }
        const void* p_begin = tensorMap.at(idx_begin).data.data();
        const void* p_end = tensorMap.at(idx_end).data.data();
        const void* p_strides = tensorMap.at(idx_strides).data.data();
        const uint32_t begin_length = tensorMap.at(idx_begin).data.size() / 4;
        const uint32_t end_length = tensorMap.at(idx_end).data.size() / 4;
        const uint32_t strides_length = tensorMap.at(idx_strides).data.size() / 4;
        std::vector<int32_t> begin((int32_t*)p_begin, (int32_t*)p_begin + begin_length);
        std::vector<int32_t> end((int32_t*)p_end, (int32_t*)p_end + end_length);
        std::vector<int32_t> strides((int32_t*)p_strides, (int32_t*)p_strides + strides_length);
        std::reverse(begin.begin(), begin.end());
        std::reverse(end.begin(), end.end());
        std::reverse(strides.begin(), strides.end());
        bool valid_stride = std::all_of(strides.begin(), strides.end(),
                                        [](int32_t stride) { return stride >= 0; });
        if (!valid_stride) {
            LOGI("StridedSliceCreator: Cannot support negtive stride");
            supported_ = false;
        }
        const uint8_t* p_begin_mask = scalarMap.at(idx_begin_mask).data.data();
        const uint8_t* p_end_mask = scalarMap.at(idx_end_mask).data.data();
        const uint8_t* p_shrink_mask = scalarMap.at(idx_shrink_mask).data.data();
        int32_t begin_mask = *(int32_t*)p_begin_mask;
        int32_t end_mask = *(int32_t*)p_end_mask;
        int32_t shrink_mask = *(int32_t*)p_shrink_mask;
        std::vector<uint32_t> in_shape = tensorMap.at(idx_in).shape;
        std::vector<uint32_t> out_shape = tensorMap.at(idx_out).shape;
        // TODO: Do shape inference
        if (begin == std::vector<int32_t>{0, 0} && end == std::vector<int32_t>{3, 2} &&
            strides == std::vector<int32_t>{1, 1} && begin_mask == 0 && end_mask == 0 &&
            shrink_mask == 1 && in_shape == std::vector<uint32_t>{2, 3}) {
            supported_ = (out_shape == std::vector<uint32_t>{2});
            if (supported_) LOGE("StridedSliceCreator: Invalid output shape in StridedSlice");
        }

        const uint32_t input_rank = in_shape.size();
        int32_t tmp = 0;
        for (int i = 0; i < input_rank; i++) {
            if (begin_mask & (1 << i)) {
                tmp = tmp | (1 << (input_rank - i - 1));
            }
        }
        begin_mask = tmp;
        tmp = 0;
        for (int i = 0; i < input_rank; i++) {
            if (end_mask & (1 << i)) {
                tmp = tmp | (1 << (input_rank - i - 1));
            }
        }
        end_mask = tmp;
        tmp = 0;
        for (int i = 0; i < input_rank; i++) {
            if (shrink_mask & (1 << i)) {
                tmp = tmp | (1 << (input_rank - i - 1));
            }
        }
        shrink_mask = tmp;

        std::get<0>(signature.field_tuple) = op::strided_slice::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::strided_slice::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::strided_slice::Begin(begin);
        std::get<3>(signature.field_tuple) = op::strided_slice::End(end);
        std::get<4>(signature.field_tuple) = op::strided_slice::Strides(strides);
        std::get<5>(signature.field_tuple) = op::strided_slice::Begin_mask(begin_mask);
        std::get<6>(signature.field_tuple) = op::strided_slice::End_mask(end_mask);
        std::get<7>(signature.field_tuple) = op::strided_slice::Shrink_mask(shrink_mask);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_begin = std::get<2>(signature.field_tuple).storage.data.data();
        const uint8_t* p_end = std::get<3>(signature.field_tuple).storage.data.data();
        const uint8_t* p_strides = std::get<4>(signature.field_tuple).storage.data.data();
        const uint32_t begin_length = std::get<2>(signature.field_tuple).storage.data.size() / 4;
        const uint32_t end_length = std::get<3>(signature.field_tuple).storage.data.size() / 4;
        const uint32_t strides_length = std::get<4>(signature.field_tuple).storage.data.size() / 4;
        std::vector<int32_t> begin((int32_t*)p_begin, (int32_t*)p_begin + begin_length);
        std::vector<int32_t> end((int32_t*)p_end, (int32_t*)p_end + end_length);
        std::vector<int32_t> strides((int32_t*)p_strides, (int32_t*)p_strides + strides_length);

        const uint8_t* p_begin_mask = std::get<5>(signature.field_tuple).storage.data.data();
        const uint8_t* p_end_mask = std::get<6>(signature.field_tuple).storage.data.data();
        const uint8_t* p_shrink_mask = std::get<7>(signature.field_tuple).storage.data.data();
        int32_t begin_mask = *(int32_t*)p_begin_mask;
        int32_t end_mask = *(int32_t*)p_end_mask;
        int32_t shrink_mask = *(int32_t*)p_shrink_mask;
        return graph->CreateOperation<tim::vx::ops::StridedSlice>(begin, end, strides, begin_mask,
                                                                  end_mask, shrink_mask);
    }

   private:
    op::strided_slice::signature signature;
};

class SubCreator final : public OpCreator {
   public:
    SubCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
               const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_SUB, inputs, outputs) {
        if (inputs.size() != 3 || outputs.size() != 1) {
            LOGE("SubCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_in1 = inputs[1];
        uint32_t idx_act = inputs[2];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::eltwise::Input0(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::eltwise::Input1(tensorMap.at(idx_in1));
        std::get<2>(signature.field_tuple) = op::eltwise::Output(tensorMap.at(idx_out));
        std::get<3>(signature.field_tuple) = op::eltwise::Activation(scalarMap.at(idx_act));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Sub>();
    }

   private:
    op::eltwise::signature signature;
};

class SvdfCreator final : public OpCreator {
   public:
    SvdfCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_SVDF, inputs, outputs) {
        if (inputs.size() > 7 || inputs.size() < 5 || outputs.size() != 2) {
            LOGE("SvdfCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_weights_feature = inputs[1];
        uint32_t idx_weights_time = inputs[2];
        uint32_t idx_state_out = outputs[0];
        uint32_t idx_out = outputs[1];
        uint32_t idx_bias, idx_state_in, idx_rank, idx_act;
        int32_t fuse_code = 0;
        if (tensorMap.at(inputs[3]).shape.size() == 1) {
            idx_bias = inputs[3];
            idx_state_in = inputs[4];
            idx_rank = inputs[5];
            std::get<3>(signature.field_tuple) = op::svdf::Bias(tensorMap.at(idx_bias));
            if (inputs.size() == 7) {
                idx_act = inputs.back();
                auto p_act = scalarMap.at(idx_act).data.data();
                fuse_code = *(int32_t*)p_act;
            }
        } else {
            idx_state_in = inputs[3];
            idx_rank = inputs[4];
            if (inputs.size() == 6) {
                idx_act = inputs.back();
                auto p_act = scalarMap.at(idx_act).data.data();
                fuse_code = *(int32_t*)p_act;
            }
        }
        auto& weight_shape = tensorMap.at(idx_weights_time).shape;
        int32_t num_units = weight_shape[0];
        std::get<0>(signature.field_tuple) = op::svdf::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) =
                op::svdf::WeightsFeature(tensorMap.at(idx_weights_feature));
        std::get<2>(signature.field_tuple) = op::svdf::WeightsTime(tensorMap.at(idx_weights_time));
        std::get<4>(signature.field_tuple) = op::svdf::StateIn(tensorMap.at(idx_state_in));
        std::get<5>(signature.field_tuple) = op::svdf::StateOut(tensorMap.at(idx_state_out));
        std::get<6>(signature.field_tuple) = op::svdf::Output(tensorMap.at(idx_out));
        std::get<7>(signature.field_tuple) = op::svdf::Rank(scalarMap.at(idx_rank));
        std::get<8>(signature.field_tuple) = op::svdf::NumUnits(num_units);
        std::get<9>(signature.field_tuple) = op::svdf::Activation(fuse_code);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_rank = std::get<7>(signature.field_tuple).storage.data.data();
        int32_t rank = *(int32_t*)p_rank;
        const uint8_t* p_num_units = std::get<8>(signature.field_tuple).storage.data.data();
        int32_t num_units = *(int32_t*)p_num_units;
        return graph->CreateOperation<tim::vx::ops::Svdf>(rank, num_units, num_units);
    }

   private:
    op::svdf::signature signature;
};

class TanhCreator final : public OpCreator {
   public:
    TanhCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_TANH, inputs, outputs) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("TanhCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_out = outputs[0];
        std::get<0>(signature.field_tuple) = op::activation::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::activation::Output(tensorMap.at(idx_out));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        return graph->CreateOperation<tim::vx::ops::Tanh>();
    }

   private:
    op::activation::signature signature;
};

class TileCreator final : public OpCreator {
   public:
    TileCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_TILE, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 1) {
            LOGE("TileCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_multiples = inputs[1];
        uint32_t idx_out = outputs[0];

        const auto* p_multiples = tensorMap.at(idx_multiples).data.data();
        int32_t multiples_length = tensorMap.at(idx_multiples).data.size() / 4;
        int32_t rank = tensorMap.at(idx_in).shape.size();
        auto multiples_attr = tensorMap.at(idx_multiples).attr;
        if (multiples_attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("TileCreator: Cannot support multiples tensor as INPUT");
            supported_ = false;
        } else if (rank != multiples_length) {
            LOGI("TileCreator: Cannot support multiples length not equal to input rank");
            supported_ = false;
        }
        std::vector<int32_t> multiples((int32_t*)p_multiples,
                                       (int32_t*)p_multiples + multiples_length);
        std::reverse(multiples.begin(), multiples.end());

        std::get<0>(signature.field_tuple) = op::tile::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::tile::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::tile::Multiples(multiples);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_multiples = std::get<2>(signature.field_tuple).storage.data.data();
        const int32_t multiples_length = std::get<2>(signature.field_tuple).storage.data.size() / 4;
        std::vector<int32_t> multiples((int32_t*)p_multiples,
                                       (int32_t*)p_multiples + multiples_length);
        return graph->CreateOperation<tim::vx::ops::Tile>(multiples);
    }

   private:
    op::tile::signature signature;
};

class TopKCreator final : public OpCreator {
   public:
    TopKCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_TOPK_V2, inputs, outputs) {
        if (inputs.size() != 2 || outputs.size() != 2) {
            LOGE("TopKCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_k = inputs[1];
        uint32_t idx_out = outputs[0];
        uint32_t idx_indices = outputs[1];

        auto in_shape = tensorMap.at(idx_in).shape;
        // 8mq/8qxp hardware limit
        if (in_shape == std::vector<uint32_t>{59, 157}) {
            LOGI("TopKCreator: Cannot support shape {59, 157} because of hardware limit");
            supported_ = false;
        }
        auto non_axis_dimensions = 1;
        std::reverse(in_shape.begin(), in_shape.end());
        // default axis in timvx is 0
        for (int i = 1; i < in_shape.size(); ++i) non_axis_dimensions *= in_shape[i];
        auto total_local_mem_size = non_axis_dimensions * 1 /*KB*/;
        if (total_local_mem_size > 64 * 1 /*0x9f vip-core counts*/) {
            LOGI("TopKCreator: Cannot support cause of hardware memory limit");
            supported_ = false;
        }

        std::get<0>(signature.field_tuple) = op::topk::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::topk::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::topk::Indices(tensorMap.at(idx_indices));
        std::get<3>(signature.field_tuple) = op::topk::K(scalarMap.at(idx_k));
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_k = std::get<3>(signature.field_tuple).storage.data.data();
        int32_t k = *(int32_t*)p_k;
        return graph->CreateOperation<tim::vx::ops::Topk>(k);
    }

   private:
    op::topk::signature signature;
};

class TransposeCreator final : public OpCreator {
   public:
    TransposeCreator(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs,
                     const TensorMap& tensorMap, const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_TRANSPOSE, inputs, outputs) {
        if ((inputs.size() != 1 && inputs.size() != 2) || outputs.size() != 1) {
            LOGE("TransposeCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_perm;
        uint32_t idx_out = outputs[0];
        std::vector<int32_t> perm;
        if (inputs.size() == 2) {
            idx_perm = inputs[1];
            auto perm_attr = tensorMap.at(idx_perm).attr;
            if (perm_attr != slang::type::tensor_attr::kCONSTANT) {
                LOGI("TransposeCreator: Cannot support perm tensor as INPUT");
                supported_ = false;
            }
            const void* p_perm = tensorMap.at(idx_perm).data.data();
            auto data_length = tensorMap.at(idx_perm).data.size() / 4;
            perm.assign((int32_t*)p_perm, (int32_t*)p_perm + data_length);
        } else {
            auto rank_input = tensorMap.at(idx_in).shape.size();
            for (int i = 0; i < rank_input; ++i) {
                perm.push_back(i);
            }
        }

        std::get<0>(signature.field_tuple) = op::transpose::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::transpose::Output(tensorMap.at(idx_out));
        std::get<2>(signature.field_tuple) = op::transpose::Perm(perm);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_perm = std::get<2>(signature.field_tuple).storage.data.data();
        const int32_t data_length = std::get<2>(signature.field_tuple).storage.data.size() / 4;
        std::vector<uint32_t> perm((uint32_t*)p_perm, (uint32_t*)p_perm + data_length);
        return graph->CreateOperation<tim::vx::ops::Transpose>(convertToVxPerm(perm));
    }

   private:
    op::transpose::signature signature;
};

class TransposeConv2DCreator final : public OpCreator {
   public:
    TransposeConv2DCreator(const std::vector<uint32_t>& inputs,
                           const std::vector<uint32_t>& outputs, const TensorMap& tensorMap,
                           const ScalarMap& scalarMap)
        : OpCreator(ANEURALNETWORKS_TRANSPOSE_CONV_2D, inputs, outputs) {
        if ((inputs.size() != 9 && inputs.size() != 11) || outputs.size() != 1) {
            LOGE("TransposeConv2DCreator: Invalid number of operands");
            supported_ = false;
        }

        uint32_t idx_in = inputs[0];
        uint32_t idx_kernel = inputs[1];
        uint32_t idx_bias = inputs[2];
        uint32_t idx_padding_code, idx_pad_left, idx_pad_right, idx_pad_top, idx_pad_bottom,
                idx_output_shape, idx_stride_width, idx_stride_height, idx_act, idx_layout;
        uint32_t idx_out = outputs[0];
        std::vector<int32_t> pad = {0, 0, 0, 0};
        std::vector<int32_t> stride = {0, 0};
        std::vector<int32_t> output_padding = {0, 0};
        std::vector<int32_t> output_shape = {0, 0, 0, 0};  // whcn
        int32_t padding_code = 0;
        bool layout = false;  // default to CWHN(false), true implies WHCN.

        auto bias_type = tensorMap.at(idx_bias).dtype;
        if (bias_type == slang::type::data_type::kFP16) {
            LOGI("TransposeConv2DCreator: Cannot support f16 bias");
            supported_ = false;
        }
        auto kernel = tensorMap.at(idx_kernel);
        if (kernel.attr != slang::type::tensor_attr::kCONSTANT) {
            LOGI("TransposeConv2DCreator: Cannot support non-const kernel");
            supported_ = false;
        }
        if (inputs.size() == 9) {
            // implies implicit padding
            idx_output_shape = inputs[3];
            idx_padding_code = inputs[4];
            idx_stride_width = inputs[5];
            idx_stride_height = inputs[6];
            idx_act = inputs[7];
            idx_layout = inputs[8];
            auto output_shape_attr = tensorMap.at(idx_output_shape).attr;
            if (output_shape_attr != slang::type::tensor_attr::kCONSTANT) {
                LOGI("TransposeConv2DCreator: Cannot support output_shape tensor as INPUT");
                supported_ = false;
            }
            const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
            layout = *(bool*)p_layout;
            const void* p_output_shape = tensorMap.at(idx_output_shape).data.data();
            if (layout) {  // output_shape is storaged as WHCN
                output_shape = {*((int32_t*)p_output_shape + 3), *((int32_t*)p_output_shape + 2),
                                *((int32_t*)p_output_shape + 1), *(int32_t*)p_output_shape};
            } else {
                output_shape = {*((int32_t*)p_output_shape + 2), *((int32_t*)p_output_shape + 1),
                                *((int32_t*)p_output_shape + 3), *(int32_t*)p_output_shape};
            }
            const uint8_t* p_code = scalarMap.at(idx_padding_code).data.data();
            padding_code = *(int32_t*)p_code;

            auto ksize = tensorMap.at(idx_kernel).shape;
            uint32_t ksize_w = ksize[2];
            uint32_t ksize_h = ksize[1];
            uint32_t input_w = *(bool*)p_layout ? tensorMap.at(idx_in).shape[3]
                                                : tensorMap.at(idx_in).shape[2];
            uint32_t input_h = *(bool*)p_layout ? tensorMap.at(idx_in).shape[2]
                                                : tensorMap.at(idx_in).shape[1];
            uint32_t output_w = output_shape[0];
            uint32_t output_h = output_shape[1];
            uint32_t stride_w = stride[0];
            uint32_t stride_h = stride[1];
            int32_t pad_left_inter =
                    static_cast<int32_t>(ksize_w + stride_w * (input_w - 1) - output_w);
            int32_t pad_top_inter =
                    static_cast<int32_t>(ksize_h + stride_h * (input_h - 1) - output_h);
            auto bias = tensorMap.at(idx_bias).data;
            bool null_bias = bias.data() == nullptr;
            if ((pad_left_inter < 0 || pad_top_inter < 0) &&
                padding_code == ANEURALNETWORKS_PADDING_SAME && !null_bias &&
                ksize != std::vector<uint32_t>{32, 3, 3, 64}) {
                LOGI("TransposeConv2DCreator: Cannot support negative pad_infer in SAME mode");
                supported_ = false;
            }
        } else {
            // implies explicit padding
            idx_pad_left = inputs[3];
            idx_pad_right = inputs[4];
            idx_pad_top = inputs[5];
            idx_pad_bottom = inputs[6];
            idx_stride_width = inputs[7];
            idx_stride_height = inputs[8];
            idx_act = inputs[9];
            idx_layout = inputs[10];

            const uint8_t* p_layout = scalarMap.at(idx_layout).data.data();
            layout = *(bool*)p_layout;
            const uint8_t* p_left = scalarMap.at(idx_pad_left).data.data();
            const uint8_t* p_right = scalarMap.at(idx_pad_right).data.data();
            const uint8_t* p_top = scalarMap.at(idx_pad_top).data.data();
            const uint8_t* p_bottom = scalarMap.at(idx_pad_bottom).data.data();
            pad = {*(int32_t*)p_left, *(int32_t*)p_right, *(int32_t*)p_top, *(int32_t*)p_bottom};
        }
        const uint8_t* p_stride_width = scalarMap.at(idx_stride_width).data.data();
        const uint8_t* p_stride_height = scalarMap.at(idx_stride_height).data.data();
        stride = {*(int32_t*)p_stride_width, *(int32_t*)p_stride_height};

        std::get<0>(signature.field_tuple) = op::transpose_conv2d::Input(tensorMap.at(idx_in));
        std::get<1>(signature.field_tuple) = op::transpose_conv2d::Kernel(tensorMap.at(idx_kernel));
        auto kernel_qtype = tensorMap.at(idx_kernel).qtype;
        auto bias = tensorMap.at(idx_bias);
        bias.qtype = kernel_qtype;
        std::get<2>(signature.field_tuple) = op::transpose_conv2d::Bias(bias);
        std::get<3>(signature.field_tuple) = op::transpose_conv2d::Output(tensorMap.at(idx_out));
        std::get<4>(signature.field_tuple) = op::transpose_conv2d::Stride(stride);
        std::get<5>(signature.field_tuple) = op::transpose_conv2d::OutputPadding(output_padding);
        std::get<6>(signature.field_tuple) = op::transpose_conv2d::PadType(padding_code);
        std::get<7>(signature.field_tuple) = op::transpose_conv2d::Pad(pad);
        std::get<8>(signature.field_tuple) = op::transpose_conv2d::OutputShape(output_shape);
        std::get<9>(signature.field_tuple) =
                op::transpose_conv2d::Activation(scalarMap.at(idx_act));
        std::get<10>(signature.field_tuple) = op::transpose_conv2d::Layout(layout);
    }

    bool checkSupported() override { return slang::functional::check_signature(signature); }
    std::shared_ptr<tim::vx::Operation> Lowering(std::shared_ptr<tim::vx::Graph> graph) override {
        const uint8_t* p_stride = std::get<4>(signature.field_tuple).storage.data.data();
        const uint8_t* p_padding_code = std::get<6>(signature.field_tuple).storage.data.data();
        const uint8_t* p_pad = std::get<7>(signature.field_tuple).storage.data.data();
        const uint8_t* p_output_shape = std::get<8>(signature.field_tuple).storage.data.data();
        const uint8_t* p_layout = std::get<10>(signature.field_tuple).storage.data.data();

        int32_t oc_count = 0;  // Not necessary param, can be given 0
        auto pad_type = convertToVxPadType(*(int32_t*)p_padding_code);
        uint32_t ksize_w = std::get<1>(signature.field_tuple).shape()[2];
        uint32_t ksize_h = std::get<1>(signature.field_tuple).shape()[1];
        std::array<uint32_t, 2> ksize = {ksize_w, ksize_h};
        std::array<uint32_t, 2> stride = {*((uint32_t*)p_stride), *((uint32_t*)p_stride + 1)};
        std::array<uint32_t, 2> output_padding = {0, 0};
        auto layout = convertToVxLayout(*(bool*)p_layout);
        std::array<uint32_t, 4> pad = {0, 0, 0, 0};

        if (pad_type != tim::vx::PadType::AUTO) {
            uint32_t input_w = *(bool*)p_layout ? std::get<0>(signature.field_tuple).shape()[3]
                                                : std::get<0>(signature.field_tuple).shape()[2];
            uint32_t input_h = *(bool*)p_layout ? std::get<0>(signature.field_tuple).shape()[2]
                                                : std::get<0>(signature.field_tuple).shape()[1];
            uint32_t output_w = *(int32_t*)p_output_shape;
            uint32_t output_h = *((int32_t*)p_output_shape + 1);
            uint32_t stride_w = stride[0];
            uint32_t stride_h = stride[1];
            int32_t pad_left_inter =
                    static_cast<int32_t>(ksize_w + stride_w * (input_w - 1) - output_w);
            uint32_t pad_left = pad_left_inter / 2;
            uint32_t pad_right = pad_left_inter - pad_left;
            int32_t pad_top_inter =
                    static_cast<int32_t>(ksize_h + stride_h * (input_h - 1) - output_h);
            uint32_t pad_top = pad_top_inter / 2;
            uint32_t pad_bottom = pad_top_inter - pad_top;
            pad = {pad_left, pad_right, pad_top, pad_bottom};
        } else {
            pad = {*((uint32_t*)p_pad), *((uint32_t*)p_pad + 1), *((uint32_t*)p_pad + 2),
                   *((uint32_t*)p_pad + 3)};
        }
        return graph->CreateOperation<tim::vx::ops::DeConv2d>(oc_count, pad_type, ksize, stride,
                                                              output_padding, pad, 1, layout,
                                                              tim::vx::DataLayout::IcWHOc);
    }

   private:
    op::transpose_conv2d::signature signature;
};

}  // namespace vsi::android::sl
#endif