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
#ifndef VSI_ANDROID_SL_OP_CREATOR_H
#define VSI_ANDROID_SL_OP_CREATOR_H
#include <cassert>
#include <memory>
#include <unordered_map>
#include <algorithm>

#include "Utils.h"
#include "slang/functional.h"
#include "slang/type_system.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops.h"
#include "spec/ops/activation/spec.h"
#include "spec/ops/batch_to_space/spec.h"
#include "spec/ops/conv2d/spec.h"
#include "spec/ops/depthwise_conv2d/spec.h"
#include "spec/ops/depth_to_space/spec.h"
#include "spec/ops/dequantize/spec.h"
#include "spec/ops/eltwise/spec.h"
#include "spec/ops/eltwise_unary/spec.h"
#include "spec/ops/floor/spec.h"
#include "spec/ops/fully_connected/spec.h"
#include "spec/ops/grouped_conv2d/spec.h"
#include "spec/ops/l2_normalization/spec.h"
#include "spec/ops/logical_not/spec.h"
#include "spec/ops/logical_and_or/spec.h"
#include "spec/ops/mean/spec.h"
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
#include "spec/ops/rsqrt/spec.h"
#include "spec/ops/softmax/spec.h"
#include "spec/ops/space_to_depth/spec.h"
#include "spec/ops/space_to_batch/spec.h"
#include "spec/ops/transpose_conv2d/spec.h"

namespace vsi {
namespace android {
namespace sl {

using TensorMap = std::unordered_map<uint32_t, slang::type::tensor_storage>;
using ScalarMap = std::unordered_map<uint32_t, slang::type::scalar_storage>;

inline int32_t ConvertAxis(int32_t axisIn, uint32_t dimNum) {
  return dimNum - (axisIn < 0 ? dimNum + axisIn : axisIn) - 1;
}

inline tim::vx::PadType AndroidPadTypeToVsiPadType(int32_t padding_code) {
  switch (padding_code) {
    case 0:
      return tim::vx::PadType::AUTO;
    case ANEURALNETWORKS_PADDING_SAME:
      return tim::vx::PadType::SAME;
    case ANEURALNETWORKS_PADDING_VALID:
      return tim::vx::PadType::VALID;
    default:
      std::cout << "Unsuppoted pad type." << std::endl;
      return tim::vx::PadType::AUTO;
  }
}

inline tim::vx::DataLayout AndroidLayoutToVsiLayout(uint8_t layout_flag) {
  switch (layout_flag) {
    case 0:
      return tim::vx::DataLayout::CWHN;
    case 1:
      return tim::vx::DataLayout::WHCN;
    default:
      std::cout << "Unsuppoted layout type." << std::endl;
      return tim::vx::DataLayout::ANY;
  }
}

class OpCreator {
 public:
  OpCreator() {}
  virtual ~OpCreator() {}
  virtual bool Check() = 0;
  virtual std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) = 0;

  ANeuralNetworksOperationType Type() { return type_; }
  std::vector<uint32_t>& Inputs() { return inputs_; }
  std::vector<uint32_t>& Outputs() { return outputs_; }

 protected:
  ANeuralNetworksOperationType type_;
  std::vector<uint32_t> inputs_;
  std::vector<uint32_t> outputs_;
};

class AbsCreator : public OpCreator {
 public:
  AbsCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Abs gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_ABS;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::eltwise_unary::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::eltwise_unary::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Abs>();
  }

 private:
  op::eltwise_unary::signature signature;
};

class AddCreator : public OpCreator {
 public:
  AddCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 3 || outputs.size() != 1) {
      std::cout << "Add gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_ADD;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_act = inputs[2];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::eltwise::Input0(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::eltwise::Input1(tensor_map.at(idx_in1));
    std::get<2>(signature.field_tuple) = op::eltwise::Output(tensor_map.at(idx_out));
    std::get<3>(signature.field_tuple) = op::eltwise::Activation(scalar_map.at(idx_act));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Add>();
  }

 private:
  op::eltwise::signature signature;
};

class AveragePool2DCreator : public OpCreator {
 public:
  AveragePool2DCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if ((inputs.size() != 7 && inputs.size() != 8 && inputs.size() != 10 &&
       inputs.size() != 11) || outputs.size() != 1) {
      std::cout << "AveragePool2D gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_AVERAGE_POOL_2D;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_padding_code, idx_pad_left, idx_pad_right, idx_pad_top,
        idx_pad_bottom, idx_stride_width, idx_stride_height, idx_filter_width,
        idx_filter_height, idx_act, idx_layout;
    uint32_t idx_out = outputs[0];
    std::vector<int32_t> pad = {0, 0, 0, 0};
    std::vector<int32_t> stride = {0, 0};
    std::vector<int32_t> filter = {0, 0};
    int32_t padding_code = 0;
    bool layout_flag = false; //default to CWHN(false), true implies WHCN.

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

      const uint8_t* p_left = scalar_map.at(idx_pad_left).data.data();
      const uint8_t* p_right = scalar_map.at(idx_pad_right).data.data();
      const uint8_t* p_top = scalar_map.at(idx_pad_top).data.data();
      const uint8_t* p_bottom = scalar_map.at(idx_pad_bottom).data.data();
      pad[0] = *(int32_t*)p_left;
      pad[1] = *(int32_t*)p_right;
      pad[2] = *(int32_t*)p_top;
      pad[3] = *(int32_t*)p_bottom;

      if (inputs.size() == 11){
        idx_layout = inputs[10];
        const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
        layout_flag = *(bool*)p_layout;
      }
    }
    if (inputs.size() == 7 || inputs.size() == 8) {
      idx_padding_code = inputs[1];
      idx_stride_width = inputs[2];
      idx_stride_height = inputs[3];
      idx_filter_width = inputs[4];
      idx_filter_height = inputs[5];
      idx_act = inputs[6];

      const uint8_t* p_code = scalar_map.at(idx_padding_code).data.data();
      padding_code = *(int32_t*)p_code;

      if (inputs.size() == 8){
        idx_layout = inputs[7];
        const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
        layout_flag = *(bool*)p_layout;
      }
    }
    const uint8_t* p_stride_width = scalar_map.at(idx_stride_width).data.data();
    const uint8_t* p_stride_height = scalar_map.at(idx_stride_height).data.data();
    const uint8_t* p_filter_width = scalar_map.at(idx_filter_width).data.data();
    const uint8_t* p_filter_height = scalar_map.at(idx_filter_height).data.data();
    stride[0] = *(int32_t*)p_stride_width;
    stride[1] = *(int32_t*)p_stride_height;
    filter[0] = *(int32_t*)p_filter_width;
    filter[1] = *(int32_t*)p_filter_width;

    std::get<0>(signature.field_tuple) = op::pool2d::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::pool2d::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::pool2d::Pad(pad);//construct scalar_feild
    std::get<3>(signature.field_tuple) = op::pool2d::PaddingCode(padding_code);
    std::get<4>(signature.field_tuple) = op::pool2d::Stride(stride);
    std::get<5>(signature.field_tuple) = op::pool2d::Filter(filter);
    std::get<6>(signature.field_tuple) = op::pool2d::Activation(scalar_map.at(idx_act));
    std::get<7>(signature.field_tuple) = op::pool2d::Layout(layout_flag);
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    const uint8_t* p_pad = std::get<2>(signature.field_tuple).storage.data.data();
    const uint8_t* p_padding_code = std::get<3>(signature.field_tuple).storage.data.data();
    const uint8_t* p_stride = std::get<4>(signature.field_tuple).storage.data.data();
    const uint8_t* p_filter = std::get<5>(signature.field_tuple).storage.data.data();
    const uint8_t* p_layout = std::get<7>(signature.field_tuple).storage.data.data();
    std::array<uint32_t, 4> pad = {*(uint32_t*)p_pad, *((uint32_t*)p_pad + 1),
                                   *((uint32_t*)p_pad + 2), *((uint32_t*)p_pad + 3)};
    std::array<uint32_t, 2> stride = {*((uint32_t*)p_stride), *((uint32_t*)p_stride + 1)};
    std::array<uint32_t, 2> filter = {*((uint32_t*)p_filter), *((uint32_t*)p_filter + 1)};

    if (*(int32_t*)p_padding_code == 0) {
      return graph->CreateOperation<tim::vx::ops::Pool2d>(
        tim::vx::PoolType::AVG_ANDROID, pad, filter, stride,
        tim::vx::RoundType::FLOOR, AndroidLayoutToVsiLayout(*(bool*)p_layout));
    } else {
      return graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG_ANDROID,
        AndroidPadTypeToVsiPadType(*(int32_t*)p_padding_code), filter, stride,
        tim::vx::RoundType::FLOOR, AndroidLayoutToVsiLayout(*(bool*)p_layout));
    }
  }

 private:
  op::pool2d::signature signature;
};

class BatchToSpaceCreator : public OpCreator {
 public:
  BatchToSpaceCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if ((inputs.size() != 2 && inputs.size() != 3)|| outputs.size() != 1) {
      std::cout << "BatchToSpace gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_BATCH_TO_SPACE_ND;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_block_size = inputs[1];
    uint32_t idx_layout;
    uint32_t idx_out = outputs[0];

    auto block_size_attr = tensor_map.at(idx_block_size).attr;
    if (block_size_attr != slang::type::tensor_attr::kCONSTANT) {
      std::cout << "Block_size tensor as INPUT are not supported in BatchToSpace" << std::endl;
      assert(false);
    }
    const void* p_block_size = tensor_map.at(idx_block_size).data;
    const uint32_t block_size_length = tensor_map.at(idx_block_size).data_length / 4;
    std::vector<int32_t> block_size((int32_t*)p_block_size, (int32_t*)p_block_size + block_size_length);

    bool layout_flag = false;
    if (inputs.size() == 3) {
      idx_layout = inputs[2];
      const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
      layout_flag = *(bool*)p_layout;
    }
    std::get<0>(signature.field_tuple) = op::batch_to_space::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::batch_to_space::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::batch_to_space::Block_size(block_size);
    std::get<3>(signature.field_tuple) = op::batch_to_space::Crop(std::vector<int>{0, 0, 0, 0});
    std::get<4>(signature.field_tuple) = op::batch_to_space::Layout(layout_flag);
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
      const uint8_t* p_block_size = std::get<2>(signature.field_tuple).storage.data.data();
      const uint8_t* p_layout = std::get<4>(signature.field_tuple).storage.data.data();
      // block_size reverse as input shape reverse
      std::vector<int32_t> block_size = {*((int32_t*)p_block_size + 1), *(int32_t*)p_block_size};
      return graph->CreateOperation<tim::vx::ops::Batch2Space>(
              block_size, std::vector<int>{0, 0, 0, 0}, AndroidLayoutToVsiLayout(*(bool*)p_layout));
  }

 private:
  op::batch_to_space::signature signature;
};

class Conv2DCreator : public OpCreator {
 public:
  Conv2DCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
      if ((inputs.size() != 7 && inputs.size() != 8 && inputs.size() != 10 &&
           inputs.size() != 11 && inputs.size() != 13) ||
          outputs.size() != 1) {
          std::cout << "Conv2D gets invalid number of operands" << std::endl;
          assert(false);
      }
    type_ = ANEURALNETWORKS_CONV_2D;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_kernel = inputs[1];
    uint32_t idx_bias = inputs[2];
    uint32_t idx_padding_code, idx_pad_left, idx_pad_right, idx_pad_top, idx_pad_bottom,
            idx_stride_width, idx_stride_height, idx_act, idx_dilation_width, idx_dilation_height,
            idx_layout;
    uint32_t idx_out = outputs[0];
    std::vector<int32_t> pad = {0, 0, 0, 0};
    std::vector<int32_t> stride = {0, 0};
    std::vector<int32_t> dilation = {0, 0};
    int32_t padding_code = 0;
    bool layout_flag = false; //default to CWHN(false), true implies WHCN.

    if (inputs.size() == 7 || scalar_map.at(inputs.at(7)).dtype == slang::type::data_type::kBOOL8) {
        // implies implicit padding
        idx_padding_code = inputs[3];
        idx_stride_width = inputs[4];
        idx_stride_height = inputs[5];
        idx_act = inputs[6];
        const uint8_t* p_code = scalar_map.at(idx_padding_code).data.data();
        padding_code = *(int32_t*)p_code;
        if (inputs.size() == 8 || inputs.size() == 10){
            idx_layout = inputs[7];
            const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
            layout_flag = *(bool*)p_layout;
            if (inputs.size() == 10){
                uint32_t idx_dilation_width = inputs[8];
                uint32_t idx_dilation_height = inputs[9];
                const uint8_t* d_width = scalar_map.at(idx_dilation_width).data.data();
                const uint8_t* d_height = scalar_map.at(idx_dilation_height).data.data();
                dilation[0] = *(int32_t*)d_width;
                dilation[1] = *(int32_t*)d_height;
            }
        }
    } else { //implies explicit padding
        idx_pad_left = inputs[3];
        idx_pad_right = inputs[4];
        idx_pad_top = inputs[5];
        idx_pad_bottom = inputs[6];
        idx_stride_width = inputs[7];
        idx_stride_height = inputs[8];
        idx_act = inputs[9];

        const uint8_t* p_left = scalar_map.at(idx_pad_left).data.data();
        const uint8_t* p_right = scalar_map.at(idx_pad_right).data.data();
        const uint8_t* p_top = scalar_map.at(idx_pad_top).data.data();
        const uint8_t* p_bottom = scalar_map.at(idx_pad_bottom).data.data();
        pad[0] = *(int32_t*)p_left;
        pad[1] = *(int32_t*)p_right;
        pad[2] = *(int32_t*)p_top;
        pad[3] = *(int32_t*)p_bottom;
        if (inputs.size() == 11 || inputs.size() == 13) {
            idx_layout = inputs[10];
            const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
            layout_flag = *(bool*)p_layout;
            if (inputs.size() == 13) {
                uint32_t idx_dilation_width = inputs[11];
                uint32_t idx_dilation_height = inputs[12];
                const uint8_t* d_width = scalar_map.at(idx_dilation_width).data.data();
                const uint8_t* d_height = scalar_map.at(idx_dilation_height).data.data();
                dilation[0] = *(int32_t*)d_width;
                dilation[1] = *(int32_t*)d_height;
            }
        }
    }

    const uint8_t* p_stride_width = scalar_map.at(idx_stride_width).data.data();
    const uint8_t* p_stride_height = scalar_map.at(idx_stride_height).data.data();
    stride[0] = *(int32_t*)p_stride_width;
    stride[1] = *(int32_t*)p_stride_height;

    std::get<0>(signature.field_tuple) = op::conv2d::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::conv2d::Kernel(tensor_map.at(idx_kernel));
    auto kernel_qtype = tensor_map.at(idx_kernel).qtype;
    auto bias = tensor_map.at(idx_bias);
    bias.qtype = kernel_qtype;
    std::get<2>(signature.field_tuple) = op::conv2d::Bias(bias);
    std::get<3>(signature.field_tuple) = op::conv2d::Output(tensor_map.at(idx_out));
    std::get<4>(signature.field_tuple) = op::conv2d::Stride(stride);
    std::get<5>(signature.field_tuple) = op::conv2d::Dilation(dilation);
    std::get<6>(signature.field_tuple) = op::conv2d::PadType(padding_code);
    std::get<7>(signature.field_tuple) = op::conv2d::Pad(pad);  //construct scalar_feild
    std::get<8>(signature.field_tuple) = op::conv2d::Activation(scalar_map.at(idx_act));
    std::get<9>(signature.field_tuple) = op::conv2d::Layout(layout_flag);
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    const uint32_t* p_ksize = std::get<1>(signature.field_tuple).storage.shape.data(); //IWHO
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

    return graph->CreateOperation<tim::vx::ops::Conv2d>(
            0, AndroidPadTypeToVsiPadType(*(int32_t*)p_padding_code), ksize, stride, dilation, pad,
            0, AndroidLayoutToVsiLayout(*(bool*)p_layout), tim::vx::DataLayout::IcWHOc);
  }

 private:
  op::conv2d::signature signature;
};

class DepthwiseConv2DCreator : public OpCreator {
 public:
  DepthwiseConv2DCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
      if ((inputs.size() != 8 && inputs.size() != 9 && inputs.size() != 11 &&
           inputs.size() != 12 && inputs.size() != 14) ||
          outputs.size() != 1) {
          std::cout << "DepthwiseConv2D gets invalid number of operands" << std::endl;
          assert(false);
      }
    type_ = ANEURALNETWORKS_DEPTHWISE_CONV_2D;
    inputs_ = inputs;
    outputs_ = outputs;
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
    bool layout_flag = false; //default to CWHN(false), true implies WHCN.

    if (inputs.size() == 8 || scalar_map.at(inputs.at(8)).dtype == slang::type::data_type::kBOOL8) {
        // implies implicit padding
        idx_padding_code = inputs[3];
        idx_stride_width = inputs[4];
        idx_stride_height = inputs[5];
        idx_multipier = inputs[6];
        idx_act = inputs[7];
        const uint8_t* p_code = scalar_map.at(idx_padding_code).data.data();
        padding_code = *(int32_t*)p_code;
        if (inputs.size() == 9 || inputs.size() == 11){
            idx_layout = inputs[8];
            const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
            layout_flag = *(bool*)p_layout;
            if (inputs.size() == 11){
                uint32_t idx_dilation_width = inputs[9];
                uint32_t idx_dilation_height = inputs[10];
                const uint8_t* d_width = scalar_map.at(idx_dilation_width).data.data();
                const uint8_t* d_height = scalar_map.at(idx_dilation_height).data.data();
                dilation[0] = *(int32_t*)d_width;
                dilation[1] = *(int32_t*)d_height;
            }
        }
    } else {
      //implies explicit padding
        idx_pad_left = inputs[3];
        idx_pad_right = inputs[4];
        idx_pad_top = inputs[5];
        idx_pad_bottom = inputs[6];
        idx_stride_width = inputs[7];
        idx_stride_height = inputs[8];
        idx_multipier = inputs[9];
        idx_act = inputs[10];

        const uint8_t* p_left = scalar_map.at(idx_pad_left).data.data();
        const uint8_t* p_right = scalar_map.at(idx_pad_right).data.data();
        const uint8_t* p_top = scalar_map.at(idx_pad_top).data.data();
        const uint8_t* p_bottom = scalar_map.at(idx_pad_bottom).data.data();
        pad[0] = *(int32_t*)p_left;
        pad[1] = *(int32_t*)p_right;
        pad[2] = *(int32_t*)p_top;
        pad[3] = *(int32_t*)p_bottom;
        if (inputs.size() == 12 || inputs.size() == 14) {
            idx_layout = inputs[11];
            const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
            layout_flag = *(bool*)p_layout;
            if (inputs.size() == 14) {
                uint32_t idx_dilation_width = inputs[12];
                uint32_t idx_dilation_height = inputs[13];
                const uint8_t* d_width = scalar_map.at(idx_dilation_width).data.data();
                const uint8_t* d_height = scalar_map.at(idx_dilation_height).data.data();
                dilation[0] = *(int32_t*)d_width;
                dilation[1] = *(int32_t*)d_height;
            }
        }
    }

    const uint8_t* p_stride_width = scalar_map.at(idx_stride_width).data.data();
    const uint8_t* p_stride_height = scalar_map.at(idx_stride_height).data.data();
    stride[0] = *(int32_t*)p_stride_width;
    stride[1] = *(int32_t*)p_stride_height;

    std::get<0>(signature.field_tuple) = op::depthwise_conv2d::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::depthwise_conv2d::Kernel(tensor_map.at(idx_kernel));
    auto kernel_qtype = tensor_map.at(idx_kernel).qtype;
    auto bias = tensor_map.at(idx_bias);
    bias.qtype = kernel_qtype;
    std::get<2>(signature.field_tuple) = op::depthwise_conv2d::Bias(bias);
    std::get<3>(signature.field_tuple) = op::depthwise_conv2d::Output(tensor_map.at(idx_out));
    std::get<4>(signature.field_tuple) = op::depthwise_conv2d::Stride(stride);
    std::get<5>(signature.field_tuple) = op::depthwise_conv2d::Dilation(dilation);
    std::get<6>(signature.field_tuple) = op::depthwise_conv2d::PadType(padding_code);
    std::get<7>(signature.field_tuple) = op::depthwise_conv2d::Pad(pad);  //construct scalar_feild
    std::get<8>(signature.field_tuple) = op::depthwise_conv2d::Multiplier(scalar_map.at(idx_multipier));//construct scalar_feild
    std::get<9>(signature.field_tuple) = op::depthwise_conv2d::Activation(scalar_map.at(idx_act));
    std::get<10>(signature.field_tuple) = op::depthwise_conv2d::Layout(layout_flag);
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    const uint32_t* p_ksize = std::get<1>(signature.field_tuple).storage.shape.data(); //OWH1
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

    return graph->CreateOperation<tim::vx::ops::Conv2d>(
            0, AndroidPadTypeToVsiPadType(*(int32_t*)p_padding_code), ksize, stride, dilation, pad,
            multiplier, AndroidLayoutToVsiLayout(*(bool*)p_layout), tim::vx::DataLayout::IcWHOc);
  }

 private:
  op::depthwise_conv2d::signature signature;
};

class DepthToSpaceCreator : public OpCreator {
 public:
  DepthToSpaceCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if ((inputs.size() != 2 && inputs.size() != 3)|| outputs.size() != 1) {
      std::cout << "DepthToSpace gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_DEPTH_TO_SPACE;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_block_size = inputs[1];
    uint32_t idx_layout;
    bool layout_flag = false;
    if (inputs.size() == 3) {
      idx_layout = inputs[2];
      const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
      layout_flag = *(bool*)p_layout;
    }
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::depth_to_space::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::depth_to_space::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::depth_to_space::Block_size(scalar_map.at(idx_block_size));
    std::get<3>(signature.field_tuple) = op::depth_to_space::Layout(layout_flag);
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    const uint8_t* p_block_size = std::get<2>(signature.field_tuple).storage.data.data();
    const uint8_t* p_layout = std::get<3>(signature.field_tuple).storage.data.data();
    return graph->CreateOperation<tim::vx::ops::DepthToSpace>(
            *(int32_t*)p_block_size, tim::vx::ops::DepthToSpace::DCR_mode,
            AndroidLayoutToVsiLayout(*(bool*)p_layout));
  }

 private:
  op::depth_to_space::signature signature;
};

class DequantizeCreator : public OpCreator {
 public:
  DequantizeCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Dequantize gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_DEQUANTIZE;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::dequantize::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::dequantize::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::DataConvert>();
  }

 private:
  op::dequantize::signature signature;
};

class DivCreator : public OpCreator {
 public:
  DivCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 3 || outputs.size() != 1) {
      std::cout << "Div gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_DIV;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_act = inputs[2];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::eltwise::Input0(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::eltwise::Input1(tensor_map.at(idx_in1));
    std::get<2>(signature.field_tuple) = op::eltwise::Output(tensor_map.at(idx_out));
    std::get<3>(signature.field_tuple) = op::eltwise::Activation(scalar_map.at(idx_act));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Div>();
  }

 private:
  op::eltwise::signature signature;
};

class EluCreator : public OpCreator {
 public:
  EluCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 2 || outputs.size() != 1) {
      std::cout << "Elu gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_ELU;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_alpha = inputs[1];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::activation::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::activation::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::activation::Alpha(scalar_map.at(idx_alpha));  // Placeholder for OPTIONAL param
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    const uint8_t* p_alpha = std::get<2>(signature.field_tuple).storage.data.data();
    auto datatype = std::get<0>(signature.field_tuple).storage.dtype;
    switch (datatype)
    {
    case slang::type::data_type::kFP16:
      return graph->CreateOperation<tim::vx::ops::Elu>(*(_Float16*)p_alpha);
    default:
      return graph->CreateOperation<tim::vx::ops::Elu>(*(float*)p_alpha);
    }
  }

 private:
  op::activation::signature signature;
};

class EqualCreator : public OpCreator {
 public:
  EqualCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 2 || outputs.size() != 1) {
      std::cout << "Equal gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_EQUAL;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::relational_op::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::relational_op::Input1(tensor_map.at(idx_in1));
    std::get<2>(signature.field_tuple) = op::relational_op::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Equal>();
  }

 private:
  op::relational_op::signature signature;
};

class ExpCreator : public OpCreator {
 public:
  ExpCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Exp gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_EXP;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::eltwise_unary::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::eltwise_unary::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Exp>();
  }

 private:
  op::eltwise_unary::signature signature;
};

class FloorCreator : public OpCreator {
 public:
  FloorCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Floor gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_FLOOR;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::floor::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::floor::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Floor>();
  }

 private:
  op::floor::signature signature;
};

class FullyConnectedCreator : public OpCreator {
 public:
  FullyConnectedCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 4 || outputs.size() != 1) {
      std::cout << "FullyConnected gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_FULLY_CONNECTED;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_weight = inputs[1];
    uint32_t idx_bias = inputs[2];
    uint32_t idx_out = outputs[0];

    std::get<0>(signature.field_tuple) = op::fully_connected::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::fully_connected::Weight(tensor_map.at(idx_weight));
    std::get<2>(signature.field_tuple) = op::fully_connected::Bias(tensor_map.at(idx_bias));
    std::get<3>(signature.field_tuple) = op::fully_connected::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature) &&
           slang::functional::check_rule(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    uint32_t* p_weight = (uint32_t*)std::get<1>(signature.field_tuple).storage.shape.data();
    return graph->CreateOperation<tim::vx::ops::FullyConnected>(0, *p_weight);
  }

 private:
  op::fully_connected::signature signature;
};

class GreaterCreator : public OpCreator {
 public:
  GreaterCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 2 || outputs.size() != 1) {
      std::cout << "Greater gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_GREATER;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::relational_op::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::relational_op::Input1(tensor_map.at(idx_in1));
    std::get<2>(signature.field_tuple) = op::relational_op::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Greater>();
  }

 private:
  op::relational_op::signature signature;
};

class GreaterEqualCreator : public OpCreator {
 public:
  GreaterEqualCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 2 || outputs.size() != 1) {
      std::cout << "GreaterEqual gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_GREATER_EQUAL;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::relational_op::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::relational_op::Input1(tensor_map.at(idx_in1));
    std::get<2>(signature.field_tuple) = op::relational_op::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::GreaterOrEqual>();
  }

 private:
  op::relational_op::signature signature;
};

class GroupedConv2DCreator : public OpCreator {
 public:
  GroupedConv2DCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
      if ((inputs.size() != 9 && inputs.size() != 12) ||
          outputs.size() != 1) {
          std::cout << "GroupedConv2D gets invalid number of operands" << std::endl;
          assert(false);
      }
    type_ = ANEURALNETWORKS_GROUPED_CONV_2D;
    inputs_ = inputs;
    outputs_ = outputs;
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
    bool layout_flag = false; //default to CWHN(false), true implies WHCN.

    if (inputs.size() == 9) {
      // implies implicit padding
      idx_padding_code = inputs[3];
      idx_stride_width = inputs[4];
      idx_stride_height = inputs[5];
      idx_groups = inputs[6];
      idx_act = inputs[7];
      idx_layout = inputs[8];

      const uint8_t* p_code = scalar_map.at(idx_padding_code).data.data();
      padding_code = *(int32_t*)p_code;
    } else {
      //implies explicit padding
      idx_pad_left = inputs[3];
      idx_pad_right = inputs[4];
      idx_pad_top = inputs[5];
      idx_pad_bottom = inputs[6];
      idx_stride_width = inputs[7];
      idx_stride_height = inputs[8];
      idx_groups = inputs[9];
      idx_act = inputs[10];
      idx_layout = inputs[11];

      const uint8_t* p_left = scalar_map.at(idx_pad_left).data.data();
      const uint8_t* p_right = scalar_map.at(idx_pad_right).data.data();
      const uint8_t* p_top = scalar_map.at(idx_pad_top).data.data();
      const uint8_t* p_bottom = scalar_map.at(idx_pad_bottom).data.data();
      pad[0] = *(int32_t*)p_left;
      pad[1] = *(int32_t*)p_right;
      pad[2] = *(int32_t*)p_top;
      pad[3] = *(int32_t*)p_bottom;

    }
    const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
    layout_flag = *(bool*)p_layout;
    const uint8_t* p_stride_width = scalar_map.at(idx_stride_width).data.data();
    const uint8_t* p_stride_height = scalar_map.at(idx_stride_height).data.data();
    stride[0] = *(int32_t*)p_stride_width;
    stride[1] = *(int32_t*)p_stride_height;

    std::get<0>(signature.field_tuple) = op::grouped_conv2d::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::grouped_conv2d::Kernel(tensor_map.at(idx_kernel));
    auto kernel_qtype = tensor_map.at(idx_kernel).qtype;
    auto bias = tensor_map.at(idx_bias);
    bias.qtype = kernel_qtype;
    std::get<2>(signature.field_tuple) = op::grouped_conv2d::Bias(bias);
    std::get<3>(signature.field_tuple) = op::grouped_conv2d::Output(tensor_map.at(idx_out));
    std::get<4>(signature.field_tuple) = op::grouped_conv2d::Stride(stride);
    std::get<5>(signature.field_tuple) = op::grouped_conv2d::Dilation(dilation);
    std::get<6>(signature.field_tuple) = op::grouped_conv2d::PadType(padding_code);
    std::get<7>(signature.field_tuple) = op::grouped_conv2d::Pad(pad);  //construct scalar_feild
    std::get<8>(signature.field_tuple) = op::grouped_conv2d::Groups(scalar_map.at(idx_groups));
    std::get<9>(signature.field_tuple) = op::grouped_conv2d::Activation(scalar_map.at(idx_act));
    std::get<10>(signature.field_tuple) = op::grouped_conv2d::Layout(layout_flag);
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
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

    if (*(int32_t*)p_padding_code != 0) {
      return graph->CreateOperation<tim::vx::ops::GroupedConv2d>(
              AndroidPadTypeToVsiPadType(*(int32_t*)p_padding_code), stride, dilation, groups,
              AndroidLayoutToVsiLayout(*(bool*)p_layout), tim::vx::DataLayout::IcWHOc);
    } else {
      return graph->CreateOperation<tim::vx::ops::GroupedConv2d>(
              pad, stride, dilation, groups, AndroidLayoutToVsiLayout(*(bool*)p_layout),
              tim::vx::DataLayout::IcWHOc);
    }
  }

 private:
  op::grouped_conv2d::signature signature;
};

class HardSwishCreator : public OpCreator {
 public:
  HardSwishCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Hardswish gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_HARD_SWISH;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::activation::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::activation::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::activation::Alpha(0);  // Placeholder for OPTIONAL param
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::HardSwish>();
  }

 private:
  op::activation::signature signature;
};

class L2NormalizationCreator : public OpCreator {
 public:
  L2NormalizationCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if ((inputs.size() != 1 && inputs.size() != 2) || outputs.size() != 1) {
      std::cout << "L2Normalization gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_L2_NORMALIZATION;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];

    int32_t axis = -1;
    if(inputs.size() == 2) {
      uint32_t idx_axis = inputs[1];
      auto p_axis = scalar_map.at(idx_axis).data.data();
      axis = *(int32_t*)p_axis;
    }
    int32_t vx_axis= ConvertAxis(axis, tensor_map.at(idx_in).shape.size());
    std::get<0>(signature.field_tuple) = op::l2_normalization::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::l2_normalization::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::l2_normalization::Axis(vx_axis);
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    const uint8_t* p_axis = std::get<2>(signature.field_tuple).storage.data.data();
    return graph->CreateOperation<tim::vx::ops::L2Normalization>(*(int32_t*)p_axis);
  }

 private:
  op::l2_normalization::signature signature;
};

class LessCreator : public OpCreator {
 public:
  LessCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 2 || outputs.size() != 1) {
      std::cout << "Less gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_LESS;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::relational_op::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::relational_op::Input1(tensor_map.at(idx_in1));
    std::get<2>(signature.field_tuple) = op::relational_op::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Less>();
  }

 private:
  op::relational_op::signature signature;
};

class LessEqualCreator : public OpCreator {
 public:
  LessEqualCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 2 || outputs.size() != 1) {
      std::cout << "LessEqual gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_LESS_EQUAL;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::relational_op::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::relational_op::Input1(tensor_map.at(idx_in1));
    std::get<2>(signature.field_tuple) = op::relational_op::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::LessOrEqual>();
  }

 private:
  op::relational_op::signature signature;
};

class LogCreator : public OpCreator {
 public:
  LogCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Log gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_LOG;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::eltwise_unary::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::eltwise_unary::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Log>();
  }

 private:
  op::eltwise_unary::signature signature;
};

class LogisticCreator : public OpCreator {
 public:
  LogisticCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Logistic gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_LOGISTIC;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::activation::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::activation::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::activation::Alpha(0);  // Placeholder for OPTIONAL param
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Sigmoid>();
  }

 private:
  op::activation::signature signature;
};

class LogicalAndCreator : public OpCreator {
 public:
  LogicalAndCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 2 || outputs.size() != 1) {
      std::cout << "LogicalAnd gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_LOGICAL_AND;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::logical_and_or::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::logical_and_or::Input1(tensor_map.at(idx_in1));
    std::get<2>(signature.field_tuple) = op::logical_and_or::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::LogicalAnd>();
  }

 private:
  op::logical_and_or::signature signature;
};

class LogicalNotCreator : public OpCreator {
 public:
  LogicalNotCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "LogicalNot gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_LOGICAL_NOT;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::logical_not::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::logical_not::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::LogicalNot>();
  }

 private:
  op::logical_not::signature signature;
};

class LogicalOrCreator : public OpCreator {
 public:
  LogicalOrCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 2 || outputs.size() != 1) {
      std::cout << "LogicalOr gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_LOGICAL_OR;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::logical_and_or::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::logical_and_or::Input1(tensor_map.at(idx_in1));
    std::get<2>(signature.field_tuple) = op::logical_and_or::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::LogicalOr>();
  }

 private:
  op::logical_and_or::signature signature;
};

class L2Pool2DCreator : public OpCreator {
 public:
  L2Pool2DCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if ((inputs.size() != 7 && inputs.size() != 8 && inputs.size() != 10 &&
       inputs.size() != 11) || outputs.size() != 1) {
      std::cout << "L2Pool2D gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_L2_POOL_2D;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_padding_code, idx_pad_left, idx_pad_right, idx_pad_top,
        idx_pad_bottom, idx_stride_width, idx_stride_height, idx_filter_width,
        idx_filter_height, idx_act, idx_layout;
    uint32_t idx_out = outputs[0];
    std::vector<int32_t> pad = {0, 0, 0, 0};
    std::vector<int32_t> stride = {0, 0};
    std::vector<int32_t> filter = {0, 0};
    int32_t padding_code = 0;
    bool layout_flag = false; //default to CWHN(false), true implies WHCN.

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

      const uint8_t* p_left = scalar_map.at(idx_pad_left).data.data();
      const uint8_t* p_right = scalar_map.at(idx_pad_right).data.data();
      const uint8_t* p_top = scalar_map.at(idx_pad_top).data.data();
      const uint8_t* p_bottom = scalar_map.at(idx_pad_bottom).data.data();
      pad[0] = *(int32_t*)p_left;
      pad[1] = *(int32_t*)p_right;
      pad[2] = *(int32_t*)p_top;
      pad[3] = *(int32_t*)p_bottom;

      if (inputs.size() == 11){
        idx_layout = inputs[10];
        const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
        layout_flag = *(bool*)p_layout;
      }
    }
    if (inputs.size() == 7 || inputs.size() == 8) {
      idx_padding_code = inputs[1];
      idx_stride_width = inputs[2];
      idx_stride_height = inputs[3];
      idx_filter_width = inputs[4];
      idx_filter_height = inputs[5];
      idx_act = inputs[6];

      const uint8_t* p_code = scalar_map.at(idx_padding_code).data.data();
      padding_code = *(int32_t*)p_code;

      if (inputs.size() == 8){
        idx_layout = inputs[7];
        const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
        layout_flag = *(bool*)p_layout;
      }
    }
    const uint8_t* p_stride_width = scalar_map.at(idx_stride_width).data.data();
    const uint8_t* p_stride_height = scalar_map.at(idx_stride_height).data.data();
    const uint8_t* p_filter_width = scalar_map.at(idx_filter_width).data.data();
    const uint8_t* p_filter_height = scalar_map.at(idx_filter_height).data.data();
    stride[0] = *(int32_t*)p_stride_width;
    stride[1] = *(int32_t*)p_stride_height;
    filter[0] = *(int32_t*)p_filter_width;
    filter[1] = *(int32_t*)p_filter_width;

    std::get<0>(signature.field_tuple) = op::pool2d::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::pool2d::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::pool2d::Pad(pad);//construct scalar_feild
    std::get<3>(signature.field_tuple) = op::pool2d::PaddingCode(padding_code);
    std::get<4>(signature.field_tuple) = op::pool2d::Stride(stride);
    std::get<5>(signature.field_tuple) = op::pool2d::Filter(filter);
    std::get<6>(signature.field_tuple) = op::pool2d::Activation(scalar_map.at(idx_act));
    std::get<7>(signature.field_tuple) = op::pool2d::Layout(layout_flag);
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    const uint8_t* p_pad = std::get<2>(signature.field_tuple).storage.data.data();
    const uint8_t* p_padding_code = std::get<3>(signature.field_tuple).storage.data.data();
    const uint8_t* p_stride = std::get<4>(signature.field_tuple).storage.data.data();
    const uint8_t* p_filter = std::get<5>(signature.field_tuple).storage.data.data();
    const uint8_t* p_layout = std::get<7>(signature.field_tuple).storage.data.data();
    std::array<uint32_t, 4> pad = {*(uint32_t*)p_pad, *((uint32_t*)p_pad + 1),
                                   *((uint32_t*)p_pad + 2), *((uint32_t*)p_pad + 3)};
    std::array<uint32_t, 2> stride = {*((uint32_t*)p_stride), *((uint32_t*)p_stride + 1)};
    std::array<uint32_t, 2> filter = {*((uint32_t*)p_filter), *((uint32_t*)p_filter + 1)};

    if (*(int32_t*)p_padding_code == 0) {
      return graph->CreateOperation<tim::vx::ops::Pool2d>(
        tim::vx::PoolType::L2, pad, filter, stride,
        tim::vx::RoundType::FLOOR, AndroidLayoutToVsiLayout(*(bool*)p_layout));
    } else {
      return graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::L2,
        AndroidPadTypeToVsiPadType(*(int32_t*)p_padding_code), filter, stride,
        tim::vx::RoundType::FLOOR, AndroidLayoutToVsiLayout(*(bool*)p_layout));
    }
  }

 private:
  op::pool2d::signature signature;
};

class MaxPool2DCreator : public OpCreator {
 public:
  MaxPool2DCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if ((inputs.size() != 7 && inputs.size() != 8 && inputs.size() != 10 &&
       inputs.size() != 11) || outputs.size() != 1) {
      std::cout << "MaxPool2D gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_MAX_POOL_2D;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_padding_code, idx_pad_left, idx_pad_right, idx_pad_top,
        idx_pad_bottom, idx_stride_width, idx_stride_height, idx_filter_width,
        idx_filter_height, idx_act, idx_layout;
    uint32_t idx_out = outputs[0];
    std::vector<int32_t> pad = {0, 0, 0, 0};
    std::vector<int32_t> stride = {0, 0};
    std::vector<int32_t> filter = {0, 0};
    int32_t padding_code = 0;
    bool layout_flag = false; //default to CWHN(false), true implies WHCN.

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

      const uint8_t* p_left = scalar_map.at(idx_pad_left).data.data();
      const uint8_t* p_right = scalar_map.at(idx_pad_right).data.data();
      const uint8_t* p_top = scalar_map.at(idx_pad_top).data.data();
      const uint8_t* p_bottom = scalar_map.at(idx_pad_bottom).data.data();
      pad[0] = *(int32_t*)p_left;
      pad[1] = *(int32_t*)p_right;
      pad[2] = *(int32_t*)p_top;
      pad[3] = *(int32_t*)p_bottom;

      if (inputs.size() == 11){
        idx_layout = inputs[10];
        const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
        layout_flag = *(bool*)p_layout;
      }
    }
    if (inputs.size() == 7 || inputs.size() == 8) {
      idx_padding_code = inputs[1];
      idx_stride_width = inputs[2];
      idx_stride_height = inputs[3];
      idx_filter_width = inputs[4];
      idx_filter_height = inputs[5];
      idx_act = inputs[6];

      const uint8_t* p_code = scalar_map.at(idx_padding_code).data.data();
      padding_code = *(int32_t*)p_code;

      if (inputs.size() == 8){
        idx_layout = inputs[7];
        const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
        layout_flag = *(bool*)p_layout;
      }
    }
    const uint8_t* p_stride_width = scalar_map.at(idx_stride_width).data.data();
    const uint8_t* p_stride_height = scalar_map.at(idx_stride_height).data.data();
    const uint8_t* p_filter_width = scalar_map.at(idx_filter_width).data.data();
    const uint8_t* p_filter_height = scalar_map.at(idx_filter_height).data.data();
    stride[0] = *(int32_t*)p_stride_width;
    stride[1] = *(int32_t*)p_stride_height;
    filter[0] = *(int32_t*)p_filter_width;
    filter[1] = *(int32_t*)p_filter_width;

    std::get<0>(signature.field_tuple) = op::pool2d::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::pool2d::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::pool2d::Pad(pad);//construct scalar_feild
    std::get<3>(signature.field_tuple) = op::pool2d::PaddingCode(padding_code);
    std::get<4>(signature.field_tuple) = op::pool2d::Stride(stride);
    std::get<5>(signature.field_tuple) = op::pool2d::Filter(filter);
    std::get<6>(signature.field_tuple) = op::pool2d::Activation(scalar_map.at(idx_act));
    std::get<7>(signature.field_tuple) = op::pool2d::Layout(layout_flag);
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    const uint8_t* p_pad = std::get<2>(signature.field_tuple).storage.data.data();
    const uint8_t* p_padding_code = std::get<3>(signature.field_tuple).storage.data.data();
    const uint8_t* p_stride = std::get<4>(signature.field_tuple).storage.data.data();
    const uint8_t* p_filter = std::get<5>(signature.field_tuple).storage.data.data();
    const uint8_t* p_layout = std::get<7>(signature.field_tuple).storage.data.data();
    std::array<uint32_t, 4> pad = {*(uint32_t*)p_pad, *((uint32_t*)p_pad + 1),
                                   *((uint32_t*)p_pad + 2), *((uint32_t*)p_pad + 3)};
    std::array<uint32_t, 2> stride = {*((uint32_t*)p_stride), *((uint32_t*)p_stride + 1)};
    std::array<uint32_t, 2> filter = {*((uint32_t*)p_filter), *((uint32_t*)p_filter + 1)};

    if (*(int32_t*)p_padding_code == 0) {
      return graph->CreateOperation<tim::vx::ops::Pool2d>(
        tim::vx::PoolType::MAX, pad, filter, stride,
        tim::vx::RoundType::FLOOR, AndroidLayoutToVsiLayout(*(bool*)p_layout));
    } else {
      return graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::MAX,
        AndroidPadTypeToVsiPadType(*(int32_t*)p_padding_code), filter, stride,
        tim::vx::RoundType::FLOOR, AndroidLayoutToVsiLayout(*(bool*)p_layout));
    }
  }

 private:
  op::pool2d::signature signature;
};

class MeanCreator : public OpCreator {
 public:
  MeanCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 3 || outputs.size() != 1) {
      std::cout << "Mean gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_MEAN;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_axis = inputs[1];
    uint32_t idx_keepdims = inputs[2];
    uint32_t idx_out = outputs[0];

    auto axis_attr = tensor_map.at(idx_axis).attr;
    if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
      std::cout << "Pad tensor as INPUT are not supported in Mean" << std::endl;
      assert(false);
    }
    auto p_keepdims = (bool*)scalar_map.at(idx_keepdims).data.data();
    bool keepdims = *p_keepdims;

    std::get<0>(signature.field_tuple) = op::mean::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::mean::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::mean::Axis(tensor_map.at(idx_axis));
    std::get<3>(signature.field_tuple) = op::mean::Keep_dims(keepdims);
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    std::vector<int32_t> axis_vx;
    const uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
    const void* p_axis = std::get<2>(signature.field_tuple).storage.data;
    const uint32_t data_length = std::get<2>(signature.field_tuple).storage.data_length / 4;
    for (int i = 0; i < data_length; i++) {
        int32_t axis_android = *((int32_t*)p_axis + i);
        axis_vx.push_back(ConvertAxis(axis_android, rank));
    }
    const uint8_t* p_keepdims = std::get<3>(signature.field_tuple).storage.data.data();
    const bool keepdims = *(bool*)p_keepdims;

    return graph->CreateOperation<tim::vx::ops::ReduceMean>(axis_vx, keepdims);
  }

 private:
  op::mean::signature signature;
};

class MulCreator : public OpCreator {
 public:
  MulCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 3 || outputs.size() != 1) {
      std::cout << "Mul gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_MUL;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_act = inputs[2];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::eltwise::Input0(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::eltwise::Input1(tensor_map.at(idx_in1));
    std::get<2>(signature.field_tuple) = op::eltwise::Output(tensor_map.at(idx_out));
    std::get<3>(signature.field_tuple) = op::eltwise::Activation(scalar_map.at(idx_act));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Multiply>();
  }

 private:
  op::eltwise::signature signature;
};

class NegCreator : public OpCreator {
 public:
  NegCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Neg gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_NEG;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::eltwise_unary::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::eltwise_unary::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Neg>();
  }

 private:
  op::eltwise_unary::signature signature;
};

class NotEqualCreator : public OpCreator {
 public:
  NotEqualCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 2 || outputs.size() != 1) {
      std::cout << "NotEqual gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_NOT_EQUAL;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::relational_op::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::relational_op::Input1(tensor_map.at(idx_in1));
    std::get<2>(signature.field_tuple) = op::relational_op::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::NotEqual>();
  }

 private:
  op::relational_op::signature signature;
};

class PadCreator : public OpCreator {
 public:
  PadCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if ((inputs.size() != 2) || outputs.size() != 1) {
      std::cout << "Pad gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_PAD;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_in_pad = inputs[1];
    uint32_t idx_out = outputs[0];

    auto pad_attr = tensor_map.at(idx_in_pad).attr;
    if (pad_attr != slang::type::tensor_attr::kCONSTANT) {
      std::cout << "Pad tensor as INPUT are not supported in Pad" << std::endl;
      assert(false);
    }
    std::vector<int32_t> pad;
    auto p_pad = (int32_t*)tensor_map.at(idx_in_pad).data;
    uint32_t data_length = tensor_map.at(idx_in_pad).data_length / 4;
    for (int i = 0; i < data_length; ++i) {
        pad.push_back(*(p_pad + i));
    }

    std::get<0>(signature.field_tuple) = op::pad::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::pad::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::pad::Pad(pad);
  }

  bool Check() final {
    return slang::functional::check_signature(signature) &&
           slang::functional::check_rule(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {

    uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
    auto p_pad = (uint32_t*)std::get<2>(signature.field_tuple).storage.data.data();
    std::vector<uint32_t> front_size, back_size;
    for (int i = 0; i < rank; ++i) {
        front_size.push_back(*(p_pad + i * 2));
        back_size.push_back(*(p_pad + i * 2 + 1));
    }
    //The dim value reverses along with the shape value.
    std::reverse(front_size.begin(), front_size.end());
    std::reverse(back_size.begin(), back_size.end());

    return graph->CreateOperation<tim::vx::ops::Pad>(front_size, back_size, 0);
  }

 private:
  op::pad::signature signature;
};

class PadV2Creator : public OpCreator {
 public:
  PadV2Creator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if ((inputs.size() != 3) || outputs.size() != 1) {
      std::cout << "PadV2 gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_PAD_V2;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_in_pad = inputs[1];
    uint32_t idx_const_val = inputs[2];
    uint32_t idx_out = outputs[0];

    auto pad_attr = tensor_map.at(idx_in_pad).attr;
    if (pad_attr != slang::type::tensor_attr::kCONSTANT) {
      std::cout << "Pad tensor as INPUT are not supported in Pad" << std::endl;
      assert(false);
    }
    std::vector<int32_t> pad;
    auto p_pad = (int32_t*)tensor_map.at(idx_in_pad).data;
    uint32_t data_length = tensor_map.at(idx_in_pad).data_length / 4;
    for (int i = 0; i < data_length; ++i) {
        pad.push_back(*(p_pad + i));
    }

    std::get<0>(signature.field_tuple) = op::pad_v2::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::pad_v2::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::pad_v2::Pad(pad);
    std::get<3>(signature.field_tuple) = op::pad_v2::Const_val(scalar_map.at(idx_const_val));
  }

  bool Check() final {
    return slang::functional::check_signature(signature) &&
           slang::functional::check_rule(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {

    uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
    uint32_t* p_pad = (uint32_t*)std::get<2>(signature.field_tuple).storage.data.data();
    uint8_t* p_const_val = std::get<3>(signature.field_tuple).storage.data.data();

    std::vector<uint32_t> front_size, back_size;
    for (int i = 0; i < rank; ++i) {
        front_size.push_back(*(p_pad + i * 2));
        back_size.push_back(*(p_pad + i * 2 + 1));
    }
    //The dim value reverses along with the shape value.
    std::reverse(front_size.begin(), front_size.end());
    std::reverse(back_size.begin(), back_size.end());
    auto datatype = std::get<0>(signature.field_tuple).storage.dtype;
    switch (datatype)
    {
    case slang::type::data_type::kFP16:
      return graph->CreateOperation<tim::vx::ops::PadV2>(front_size, back_size, *(_Float16*)p_const_val);
    case slang::type::data_type::kFP32:
      return graph->CreateOperation<tim::vx::ops::PadV2>(front_size, back_size, *(float*)p_const_val);
    default:
        return graph->CreateOperation<tim::vx::ops::Pad>(front_size, back_size, *(int32_t*)p_const_val);
    }
  }

 private:
  op::pad_v2::signature signature;
};

class PowCreator : public OpCreator {
 public:
  PowCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 2 || outputs.size() != 1) {
      std::cout << "Pow gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_POW;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::pow::Input0(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::pow::Input1(tensor_map.at(idx_in1));
    std::get<2>(signature.field_tuple) = op::pow::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Pow>();
  }

 private:
  op::pow::signature signature;
};

class PreluCreator : public OpCreator {
 public:
  PreluCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 2 || outputs.size() != 1) {
      std::cout << "Prelu gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_PRELU;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_alpha = inputs[1];
    uint32_t idx_out = outputs[0];

    auto alpha_attr = tensor_map.at(idx_alpha).attr;
    if (alpha_attr != slang::type::tensor_attr::kCONSTANT) {
      std::cout << "Alpha tensor as INPUT are not supported in Prelu" << std::endl;
      assert(false);
    }
    std::get<0>(signature.field_tuple) = op::prelu::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::prelu::Alpha(tensor_map.at(idx_alpha));
    std::get<2>(signature.field_tuple) = op::prelu::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Prelu>(0);
  }

 private:
  op::prelu::signature signature;
};

class QuantizeCreator : public OpCreator {
 public:
  QuantizeCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Quantize gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_QUANTIZE;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::quantize::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::quantize::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::DataConvert>();
  }

 private:
  op::quantize::signature signature;
};

class ReduceAllCreator : public OpCreator {
 public:
  ReduceAllCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 3 || outputs.size() != 1) {
      std::cout << "ReduceAll gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_REDUCE_ALL;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_axis = inputs[1];
    uint32_t idx_keepdims = inputs[2];
    uint32_t idx_out = outputs[0];

    auto axis_attr = tensor_map.at(idx_axis).attr;
    if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
      std::cout << "Axis tensor as INPUT are not supported in ReduceAll" << std::endl;
      assert(false);
    }
    std::get<0>(signature.field_tuple) = op::reduce_all_any::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::reduce_all_any::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::reduce_all_any::Axis(tensor_map.at(idx_axis));
    std::get<3>(signature.field_tuple) = op::reduce_all_any::Keep_dims(scalar_map.at(idx_keepdims));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    std::vector<int32_t> axis_vx;
    const uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
    const void* p_axis = std::get<2>(signature.field_tuple).storage.data;
    const uint32_t data_length = std::get<2>(signature.field_tuple).storage.data_length / 4;

    for (int i = 0; i < data_length; i++) {
        int32_t axis_android = *((int32_t*)p_axis + i);
        axis_vx.push_back(ConvertAxis(axis_android, rank));
    }
    const uint8_t* p_keepdims = std::get<3>(signature.field_tuple).storage.data.data();
    const bool keepdims = *(bool*)p_keepdims;
    return graph->CreateOperation<tim::vx::ops::ReduceAll>(axis_vx, keepdims);
  }

 private:
  op::reduce_all_any::signature signature;
};

class ReduceAnyCreator : public OpCreator {
 public:
  ReduceAnyCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 3 || outputs.size() != 1) {
      std::cout << "ReduceAny gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_REDUCE_ANY;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_axis = inputs[1];
    uint32_t idx_keepdims = inputs[2];
    uint32_t idx_out = outputs[0];

    auto axis_attr = tensor_map.at(idx_axis).attr;
    if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
      std::cout << "Axis tensor as INPUT are not supported in ReduceAny" << std::endl;
      assert(false);
    }
    std::get<0>(signature.field_tuple) = op::reduce_all_any::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::reduce_all_any::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::reduce_all_any::Axis(tensor_map.at(idx_axis));
    std::get<3>(signature.field_tuple) = op::reduce_all_any::Keep_dims(scalar_map.at(idx_keepdims));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    std::vector<int32_t> axis_vx;
    const uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
    const void* p_axis = std::get<2>(signature.field_tuple).storage.data;
    const uint32_t data_length = std::get<2>(signature.field_tuple).storage.data_length / 4;

    for (int i = 0; i < data_length; i++) {
        int32_t axis_android = *((int32_t*)p_axis + i);
        axis_vx.push_back(ConvertAxis(axis_android, rank));
    }
    const uint8_t* p_keepdims = std::get<3>(signature.field_tuple).storage.data.data();
    const bool keepdims = *(bool*)p_keepdims;
    return graph->CreateOperation<tim::vx::ops::ReduceAny>(axis_vx, keepdims);
  }

 private:
  op::reduce_all_any::signature signature;
};

class ReduceMaxCreator : public OpCreator {
 public:
  ReduceMaxCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 3 || outputs.size() != 1) {
      std::cout << "ReduceMax gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_REDUCE_MAX;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_axis = inputs[1];
    uint32_t idx_keepdims = inputs[2];
    uint32_t idx_out = outputs[0];

    auto axis_attr = tensor_map.at(idx_axis).attr;
    if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
      std::cout << "Axis tensor as INPUT are not supported in ReduceMax" << std::endl;
      assert(false);
    }
    std::get<0>(signature.field_tuple) = op::reduce_max_min_prod_sum::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::reduce_max_min_prod_sum::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::reduce_max_min_prod_sum::Axis(tensor_map.at(idx_axis));
    std::get<3>(signature.field_tuple) = op::reduce_max_min_prod_sum::Keep_dims(scalar_map.at(idx_keepdims));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    std::vector<int32_t> axis_vx;
    const uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
    const void* p_axis = std::get<2>(signature.field_tuple).storage.data;
    const uint32_t data_length = std::get<2>(signature.field_tuple).storage.data_length / 4;

    for (int i = 0; i < data_length; i++) {
        int32_t axis_android = *((int32_t*)p_axis + i);
        axis_vx.push_back(ConvertAxis(axis_android, rank));
    }

    const uint8_t* p_keepdims = std::get<3>(signature.field_tuple).storage.data.data();
    const bool keepdims = *(bool*)p_keepdims;
    return graph->CreateOperation<tim::vx::ops::ReduceMax>(axis_vx, keepdims);
  }

 private:
  op::reduce_max_min_prod_sum::signature signature;
};

class ReduceMinCreator : public OpCreator {
 public:
  ReduceMinCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 3 || outputs.size() != 1) {
      std::cout << "ReduceMin gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_REDUCE_MIN;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_axis = inputs[1];
    uint32_t idx_keepdims = inputs[2];
    uint32_t idx_out = outputs[0];

    auto axis_attr = tensor_map.at(idx_axis).attr;
    if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
      std::cout << "Axis tensor as INPUT are not supported in ReduceMin" << std::endl;
      assert(false);
    }
    std::get<0>(signature.field_tuple) = op::reduce_max_min_prod_sum::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::reduce_max_min_prod_sum::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::reduce_max_min_prod_sum::Axis(tensor_map.at(idx_axis));
    std::get<3>(signature.field_tuple) = op::reduce_max_min_prod_sum::Keep_dims(scalar_map.at(idx_keepdims));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    std::vector<int32_t> axis_vx;
    const uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
    const void* p_axis = std::get<2>(signature.field_tuple).storage.data;
    const uint32_t data_length = std::get<2>(signature.field_tuple).storage.data_length / 4;

    for (int i = 0; i < data_length; i++) {
        int32_t axis_android = *((int32_t*)p_axis + i);
        axis_vx.push_back(ConvertAxis(axis_android, rank));
    }

    const uint8_t* p_keepdims = std::get<3>(signature.field_tuple).storage.data.data();
    const bool keepdims = *(bool*)p_keepdims;
    return graph->CreateOperation<tim::vx::ops::ReduceMin>(axis_vx, keepdims);
  }

 private:
  op::reduce_max_min_prod_sum::signature signature;
};

class ReduceProdCreator : public OpCreator {
 public:
  ReduceProdCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 3 || outputs.size() != 1) {
      std::cout << "ReduceProd gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_REDUCE_PROD;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_axis = inputs[1];
    uint32_t idx_keepdims = inputs[2];
    uint32_t idx_out = outputs[0];

    auto axis_attr = tensor_map.at(idx_axis).attr;
    if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
      std::cout << "Axis tensor as INPUT are not supported in ReduceProd" << std::endl;
      assert(false);
    }
    std::get<0>(signature.field_tuple) = op::reduce_max_min_prod_sum::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::reduce_max_min_prod_sum::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::reduce_max_min_prod_sum::Axis(tensor_map.at(idx_axis));
    std::get<3>(signature.field_tuple) = op::reduce_max_min_prod_sum::Keep_dims(scalar_map.at(idx_keepdims));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    std::vector<int32_t> axis_vx;
    const uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
    const void* p_axis = std::get<2>(signature.field_tuple).storage.data;
    const uint32_t data_length = std::get<2>(signature.field_tuple).storage.data_length / 4;

    for (int i = 0; i < data_length; i++) {
        int32_t axis_android = *((int32_t*)p_axis + i);
        axis_vx.push_back(ConvertAxis(axis_android, rank));
    }

    const uint8_t* p_keepdims = std::get<3>(signature.field_tuple).storage.data.data();
    const bool keepdims = *(bool*)p_keepdims;
    return graph->CreateOperation<tim::vx::ops::ReduceProd>(axis_vx, keepdims);
  }

 private:
  op::reduce_max_min_prod_sum::signature signature;
};

class ReduceSumCreator : public OpCreator {
 public:
  ReduceSumCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 3 || outputs.size() != 1) {
      std::cout << "ReduceSum gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_REDUCE_SUM;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_axis = inputs[1];
    uint32_t idx_keepdims = inputs[2];
    uint32_t idx_out = outputs[0];

    auto axis_attr = tensor_map.at(idx_axis).attr;
    if (axis_attr != slang::type::tensor_attr::kCONSTANT) {
      std::cout << "Axis tensor as INPUT are not supported in ReduceSum" << std::endl;
      assert(false);
    }
    std::get<0>(signature.field_tuple) = op::reduce_max_min_prod_sum::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::reduce_max_min_prod_sum::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::reduce_max_min_prod_sum::Axis(tensor_map.at(idx_axis));
    std::get<3>(signature.field_tuple) = op::reduce_max_min_prod_sum::Keep_dims(scalar_map.at(idx_keepdims));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    std::vector<int32_t> axis_vx;
    const uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
    const void* p_axis = std::get<2>(signature.field_tuple).storage.data;
    const uint32_t data_length = std::get<2>(signature.field_tuple).storage.data_length / 4;

    for (int i = 0; i < data_length; i++) {
        int32_t axis_android = *((int32_t*)p_axis + i);
        axis_vx.push_back(ConvertAxis(axis_android, rank));
    }

    const uint8_t* p_keepdims = std::get<3>(signature.field_tuple).storage.data.data();
    const bool keepdims = *(bool*)p_keepdims;
    return graph->CreateOperation<tim::vx::ops::ReduceSum>(axis_vx, keepdims);
  }

 private:
  op::reduce_max_min_prod_sum::signature signature;
};

class ReluCreator : public OpCreator {
 public:
  ReluCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Relu gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_RELU;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::activation::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::activation::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::activation::Alpha(0);  // Placeholder for OPTIONAL param
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Relu>();
  }

 private:
  op::activation::signature signature;
};

class Relu1Creator : public OpCreator {
 public:
  Relu1Creator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Relu1 gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_RELU1;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::activation::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::activation::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::activation::Alpha(0);  // Placeholder for OPTIONAL param
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Relu1>();
  }

 private:
  op::activation::signature signature;
};

class Relu6Creator : public OpCreator {
 public:
  Relu6Creator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Relu6 gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_RELU6;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::activation::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::activation::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::activation::Alpha(0);  // Placeholder for OPTIONAL param
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Relu6>();
  }

 private:
  op::activation::signature signature;
};

class ReshapeCreator : public OpCreator {
 public:
  ReshapeCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 2 || outputs.size() != 1) {
      std::cout << "Reshape gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_RESHAPE;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_shape = inputs[1];
    uint32_t idx_out = outputs[0];

    auto shape_attr = tensor_map.at(idx_shape).attr;
    if (shape_attr != slang::type::tensor_attr::kCONSTANT) {
      std::cout << "Shape tensor as INPUT are not supported in Reshape" << std::endl;
      assert(false);
    }
    std::get<0>(signature.field_tuple) = op::reshape::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::reshape::Shape(tensor_map.at(idx_shape));
    std::get<2>(signature.field_tuple) = op::reshape::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    auto shape_tensor = std::get<1>(signature.field_tuple);
    const void* data = shape_tensor.data();
    uint32_t length = shape_tensor.data_length() / 4;  // The type of shape tensor is int32

    std::vector<int32_t> shape(length);
    for(int i=0; i<shape.size(); ++i){
      shape[i] = *((int32_t*)data + i);
    } // Get shape data
    std::vector<uint32_t> no_negative_shape;
    uint32_t total_size = 1, negative_index = 0;
    bool do_shape_inference = false;

    for (uint32_t i = 0; i < shape_tensor.shape().size(); ++i) {
      total_size *= shape_tensor.shape().at(i);
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

class RsqrtCreator : public OpCreator {
 public:
  RsqrtCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Rsqrt gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_RSQRT;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::rsqrt::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::rsqrt::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Rsqrt>();
  }

 private:
  op::rsqrt::signature signature;
};

class SinCreator : public OpCreator {
 public:
  SinCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Sin gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_SIN;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::eltwise_unary::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::eltwise_unary::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Sin>();
  }

 private:
  op::eltwise_unary::signature signature;
};

class SoftmaxCreator : public OpCreator {
 public:
  SoftmaxCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1) {
      std::cout << "Softmax gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_SOFTMAX;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_beta = inputs[1];
    uint32_t idx_out = outputs[0];
    uint32_t idx_axis;

    std::get<0>(signature.field_tuple) = op::softmax::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::softmax::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::softmax::Beta(scalar_map.at(idx_beta));
    std::get<3>(signature.field_tuple) = op::softmax::Axis(-1); //default is -1

    if (inputs.size() == 3) {
      idx_axis = inputs[2];
      std::get<3>(signature.field_tuple) = op::softmax::Axis(scalar_map.at(idx_axis));
    }
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    const uint32_t rank = std::get<0>(signature.field_tuple).storage.shape.size();
    auto datatype = std::get<2>(signature.field_tuple).storage.dtype;
    const uint8_t* beta = std::get<2>(signature.field_tuple).storage.data.data();
    const uint8_t* p_axis = std::get<3>(signature.field_tuple).storage.data.data();
    int32_t axis_android = *(int32_t*)p_axis;
    int32_t axis_vx = ConvertAxis(axis_android, rank);
    if (datatype == slang::type::data_type::kFP16) {
      return graph->CreateOperation<tim::vx::ops::Softmax>(*(_Float16*)beta, axis_vx);
    } else {
      return graph->CreateOperation<tim::vx::ops::Softmax>(*(float*)beta, axis_vx);
    }
  }

 private:
  op::softmax::signature signature;
};

class SpaceToDepthCreator : public OpCreator {
 public:
  SpaceToDepthCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if ((inputs.size() != 2 && inputs.size() != 3)|| outputs.size() != 1) {
      std::cout << "SpaceToDepth gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_SPACE_TO_DEPTH;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_block_size = inputs[1];
    uint32_t idx_layout;
    uint32_t idx_out = outputs[0];

    bool layout_flag = false;
    if (inputs.size() == 3) {
      idx_layout = inputs[2];
      const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
      layout_flag = *(bool*)p_layout;
    }
    std::get<0>(signature.field_tuple) = op::space_to_depth::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::space_to_depth::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::space_to_depth::Block_size(scalar_map.at(idx_block_size));
    std::get<3>(signature.field_tuple) = op::space_to_depth::Layout(layout_flag);
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    const uint8_t* p_block_size = std::get<2>(signature.field_tuple).storage.data.data();
    const uint8_t* p_layout = std::get<3>(signature.field_tuple).storage.data.data();
    std::vector<int32_t> block_size = {*(int32_t*)p_block_size, *(int32_t*)p_block_size};
    return graph->CreateOperation<tim::vx::ops::SpaceToDepth>(
            block_size, AndroidLayoutToVsiLayout(*(bool*)p_layout));
  }

 private:
  op::space_to_depth::signature signature;
};

class SpaceToBatchCreator : public OpCreator {
 public:
  SpaceToBatchCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if ((inputs.size() != 3 && inputs.size() != 4)|| outputs.size() != 1) {
      std::cout << "SpaceToBatch gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_SPACE_TO_BATCH_ND;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_block_size = inputs[1];
    uint32_t idx_pad = inputs[2];
    uint32_t idx_layout;
    uint32_t idx_out = outputs[0];

    auto block_size_attr = tensor_map.at(idx_block_size).attr;
    auto pad_attr = tensor_map.at(idx_pad).attr;
    if (block_size_attr != slang::type::tensor_attr::kCONSTANT ||
        pad_attr != slang::type::tensor_attr::kCONSTANT) {
      std::cout << "Block_size & pad tensor as INPUT are not supported in SpaceToBatch" << std::endl;
      assert(false);
    }
    const void* p_block_size = tensor_map.at(idx_block_size).data;
    const uint32_t block_size_length = tensor_map.at(idx_block_size).data_length / 4;
    std::vector<int32_t> block_size((int32_t*)p_block_size, (int32_t*)p_block_size + block_size_length);

    const void* p_pad = tensor_map.at(idx_pad).data;
    const uint32_t pad_length = tensor_map.at(idx_pad).data_length / 4;
    std::vector<int32_t> pad((int32_t*)p_pad, (int32_t*)p_pad + pad_length);
    bool layout_flag = false;
    if (inputs.size() == 4) {
      idx_layout = inputs[3];
      const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
      layout_flag = *(bool*)p_layout;
    }
    std::get<0>(signature.field_tuple) = op::space_to_batch::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::space_to_batch::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::space_to_batch::Block_size(block_size);
    std::get<3>(signature.field_tuple) = op::space_to_batch::Pad(pad);
    std::get<4>(signature.field_tuple) = op::space_to_batch::Layout(layout_flag);
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
      const uint8_t* p_block_size = std::get<2>(signature.field_tuple).storage.data.data();
      const uint8_t* p_pad = std::get<3>(signature.field_tuple).storage.data.data();
      const uint32_t pad_length = std::get<3>(signature.field_tuple).storage.data.size() / 4;
      const uint8_t* p_layout = std::get<4>(signature.field_tuple).storage.data.data();
      // block_size reverse as input shape reverse
      std::vector<int32_t> block_size = {*((int32_t*)p_block_size + 1), *(int32_t*)p_block_size};
      std::vector<int32_t> pad((int32_t*)p_pad, (int32_t*)p_pad + pad_length);
      // Vts pad as HW, timvx pad as WH
      std::vector<int32_t> vx_pad = {pad[2], pad[3], pad[0], pad[1]};
      return graph->CreateOperation<tim::vx::ops::Space2Batch>(
              block_size, vx_pad, AndroidLayoutToVsiLayout(*(bool*)p_layout));
  }

 private:
  op::space_to_batch::signature signature;
};

class SqrtCreator : public OpCreator {
 public:
  SqrtCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Sqrt gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_SQRT;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::rsqrt::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::rsqrt::Output(tensor_map.at(idx_out));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Sqrt>();
  }

 private:
  op::rsqrt::signature signature;
};

class SubCreator : public OpCreator {
 public:
  SubCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 3 || outputs.size() != 1) {
      std::cout << "Sub gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_SUB;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_in1 = inputs[1];
    uint32_t idx_act = inputs[2];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::eltwise::Input0(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::eltwise::Input1(tensor_map.at(idx_in1));
    std::get<2>(signature.field_tuple) = op::eltwise::Output(tensor_map.at(idx_out));
    std::get<3>(signature.field_tuple) = op::eltwise::Activation(scalar_map.at(idx_act));
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Sub>();
  }

 private:
  op::eltwise::signature signature;
};

class TanhCreator : public OpCreator {
 public:
  TanhCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::cout << "Tanh gets invalid number of operands" << std::endl;
      assert(false);
    }
    type_ = ANEURALNETWORKS_TANH;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_out = outputs[0];
    std::get<0>(signature.field_tuple) = op::activation::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::activation::Output(tensor_map.at(idx_out));
    std::get<2>(signature.field_tuple) = op::activation::Alpha(0);  // Placeholder for OPTIONAL param
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    return graph->CreateOperation<tim::vx::ops::Tanh>();
  }

 private:
  op::activation::signature signature;
};

class TransposeConv2DCreator : public OpCreator {
 public:
  TransposeConv2DCreator(const std::vector<uint32_t>& inputs,
             const std::vector<uint32_t>& outputs,
             const TensorMap& tensor_map,
             const ScalarMap& scalar_map) {
      if ((inputs.size() != 9 && inputs.size() != 11) ||
          outputs.size() != 1) {
          std::cout << "TransposeConv2D gets invalid number of operands" << std::endl;
          assert(false);
      }
    type_ = ANEURALNETWORKS_TRANSPOSE_CONV_2D;
    inputs_ = inputs;
    outputs_ = outputs;
    uint32_t idx_in = inputs[0];
    uint32_t idx_kernel = inputs[1];
    uint32_t idx_bias = inputs[2];
    uint32_t idx_padding_code, idx_pad_left, idx_pad_right, idx_pad_top, idx_pad_bottom,
            idx_output_shape, idx_stride_width, idx_stride_height, idx_act, idx_layout;
    uint32_t idx_out = outputs[0];
    std::vector<int32_t> pad = {0, 0, 0, 0};
    std::vector<int32_t> stride = {0, 0};
    std::vector<int32_t> output_padding = {0, 0};
    std::vector<int32_t> output_shape = {0, 0, 0, 0};
    int32_t padding_code = 0;
    bool layout_flag = false; //default to CWHN(false), true implies WHCN.

    if (inputs.size() == 9) {
      // implies implicit padding
      idx_output_shape = inputs[3];
      idx_padding_code = inputs[4];
      idx_stride_width = inputs[5];
      idx_stride_height = inputs[6];
      idx_act = inputs[7];
      idx_layout = inputs[8];

      auto output_shape_attr = tensor_map.at(idx_output_shape).attr;
      if (output_shape_attr != slang::type::tensor_attr::kCONSTANT) {
        std::cout << "Output_shape tensor as INPUT are not supported in TransposeConv2D" << std::endl;
        assert(false);
      }
      const void* p_output_shape = tensor_map.at(idx_output_shape).data;
      output_shape = {*(int32_t*)p_output_shape, *((int32_t*)p_output_shape + 1),
                      *((int32_t*)p_output_shape + 2), *((int32_t*)p_output_shape + 3)};
      const uint8_t* p_code = scalar_map.at(idx_padding_code).data.data();
      padding_code = *(int32_t*)p_code;
    } else {
      //implies explicit padding
      idx_pad_left = inputs[3];
      idx_pad_right = inputs[4];
      idx_pad_top = inputs[5];
      idx_pad_bottom = inputs[6];
      idx_stride_width = inputs[7];
      idx_stride_height = inputs[8];
      idx_act = inputs[9];
      idx_layout = inputs[10];

      const uint8_t* p_left = scalar_map.at(idx_pad_left).data.data();
      const uint8_t* p_right = scalar_map.at(idx_pad_right).data.data();
      const uint8_t* p_top = scalar_map.at(idx_pad_top).data.data();
      const uint8_t* p_bottom = scalar_map.at(idx_pad_bottom).data.data();
      pad = {*(int32_t*)p_left, *(int32_t*)p_right, *(int32_t*)p_top, *(int32_t*)p_bottom};
    }
    const uint8_t* p_layout = scalar_map.at(idx_layout).data.data();
    layout_flag = *(bool*)p_layout;
    const uint8_t* p_stride_width = scalar_map.at(idx_stride_width).data.data();
    const uint8_t* p_stride_height = scalar_map.at(idx_stride_height).data.data();
    stride = {*(int32_t*)p_stride_width, *(int32_t*)p_stride_height};

    std::get<0>(signature.field_tuple) = op::transpose_conv2d::Input(tensor_map.at(idx_in));
    std::get<1>(signature.field_tuple) = op::transpose_conv2d::Kernel(tensor_map.at(idx_kernel));
    auto kernel_qtype = tensor_map.at(idx_kernel).qtype;
    auto bias = tensor_map.at(idx_bias);
    bias.qtype = kernel_qtype;
    std::get<2>(signature.field_tuple) = op::transpose_conv2d::Bias(bias);
    std::get<3>(signature.field_tuple) = op::transpose_conv2d::Output(tensor_map.at(idx_out));
    std::get<4>(signature.field_tuple) = op::transpose_conv2d::Stride(stride);
    std::get<5>(signature.field_tuple) = op::transpose_conv2d::OutputPadding(output_padding);
    std::get<6>(signature.field_tuple) = op::transpose_conv2d::PadType(padding_code);
    std::get<7>(signature.field_tuple) = op::transpose_conv2d::Pad(pad);  //construct scalar_feild
    std::get<8>(signature.field_tuple) = op::transpose_conv2d::OutputShape(output_shape);
    std::get<9>(signature.field_tuple) = op::transpose_conv2d::Activation(scalar_map.at(idx_act));
    std::get<10>(signature.field_tuple) = op::transpose_conv2d::Layout(layout_flag);
  }

  bool Check() final {
    return slang::functional::check_signature(signature);
  }
  std::shared_ptr<tim::vx::Operation>
      Lowering(std::shared_ptr<tim::vx::Graph> graph) final {
    const uint8_t* p_stride = std::get<4>(signature.field_tuple).storage.data.data();
    const uint8_t* p_padding_code = std::get<6>(signature.field_tuple).storage.data.data();
    const uint8_t* p_pad = std::get<7>(signature.field_tuple).storage.data.data();
    const uint8_t* p_output_shape = std::get<8>(signature.field_tuple).storage.data.data();
    const uint8_t* p_layout = std::get<10>(signature.field_tuple).storage.data.data();

    int32_t oc_count = 0;  // Not necessary param, can be given 0
    auto pad_type =  AndroidPadTypeToVsiPadType(*(int32_t*)p_padding_code);
    uint32_t ksize_w = std::get<1>(signature.field_tuple).shape()[2];
    uint32_t ksize_h = std::get<1>(signature.field_tuple).shape()[1];
    std::array<uint32_t, 2> ksize = {ksize_w, ksize_h};
    std::array<uint32_t, 2> stride = {*((uint32_t*)p_stride), *((uint32_t*)p_stride + 1)};
    std::array<uint32_t, 2> output_padding = {0, 0};
    auto layout = AndroidLayoutToVsiLayout(*(bool*)p_layout);
    std::array<uint32_t, 4> pad = {0, 0, 0, 0};

    if (pad_type != tim::vx::PadType::AUTO) {
      uint32_t input_w = *(bool*)p_layout ? std::get<0>(signature.field_tuple).shape()[3]
                                    : std::get<0>(signature.field_tuple).shape()[2];
      uint32_t input_h = *(bool*)p_layout ? std::get<0>(signature.field_tuple).shape()[2]
                                    : std::get<0>(signature.field_tuple).shape()[1];
      uint32_t output_w = *(bool*)p_layout ? *((int32_t*)p_output_shape + 3)
                                    : *((int32_t*)p_output_shape + 2);
      uint32_t output_h = *(bool*)p_layout ? *((int32_t*)p_output_shape + 2)
                                    : *((int32_t*)p_output_shape + 1);
      uint32_t stride_w = stride[0];
      uint32_t stride_h = stride[1];
      int32_t pad_left_inter = static_cast<int32_t>(
          ksize_w + stride_w * (input_w - 1) - output_w);
      uint32_t pad_left = pad_left_inter > 0 ? pad_left_inter / 2 : 0;
      uint32_t pad_right = pad_left_inter > 0 ? pad_left_inter - pad_left : 0;
      int32_t pad_top_inter = static_cast<int32_t>(
          ksize_h + stride_h * (input_h - 1) - output_h);
      uint32_t pad_top = pad_top_inter > 0 ? pad_top_inter / 2 : 0;
      uint32_t pad_bottom = pad_top_inter > 0 ? pad_top_inter - pad_top : 0;
      pad = {pad_left, pad_right, pad_top, pad_bottom};
    } else {
      pad = {*((uint32_t*)p_pad), *((uint32_t*)p_pad + 1), *((uint32_t*)p_pad + 2), *((uint32_t*)p_pad + 3)};
    }
    return graph->CreateOperation<tim::vx::ops::DeConv2d>(
              oc_count, pad_type, ksize, stride, output_padding, pad, 1, layout, tim::vx::DataLayout::IcWHOc);
  }

 private:
  op::transpose_conv2d::signature signature;
};

}  // namespace sl
}  // namespace android
}  // namespace vsi
#endif