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
#include <iostream>

#include "Utils.h"
namespace vsi {
namespace android {
namespace sl {

tim::vx::DataType ToTvxDataType(slang::type::data_type type) {
    switch (type) {
        case slang::type::data_type::kFP32:
            return tim::vx::DataType::FLOAT32;
        case slang::type::data_type::kFP16:
            return tim::vx::DataType::FLOAT16;
        case slang::type::data_type::kINT32:
            return tim::vx::DataType::INT32;
        case slang::type::data_type::kUINT32:
            return tim::vx::DataType::UINT32;
        case slang::type::data_type::kINT16:
            return tim::vx::DataType::INT16;
        case slang::type::data_type::kUINT16:
            return tim::vx::DataType::UINT16;
        case slang::type::data_type::kINT8:
            return tim::vx::DataType::INT8;
        case slang::type::data_type::kUINT8:
            return tim::vx::DataType::UINT8;
        case slang::type::data_type::kBOOL8:
            return tim::vx::DataType::BOOL8;
        default:
            std::cout << "Unknown data type in tim-vx." << std::endl;
    }
    return tim::vx::DataType::UNKNOWN;
}

tim::vx::QuantType ToTvxQuantType(slang::type::quant_type type) {
    switch (type) {
        case slang::type::quant_type::kASYMM:
        case slang::type::quant_type::kSYMM:
            return tim::vx::QuantType::ASYMMETRIC;
        case slang::type::quant_type::kSYMM_PCQ:
            return tim::vx::QuantType::SYMMETRIC_PER_CHANNEL;
        case slang::type::quant_type::kDFP:
            return tim::vx::QuantType::DYNAMIC_FIXED_POINT;
        default:
            return tim::vx::QuantType::NONE;
    }
    return tim::vx::QuantType::NONE;
}

slang::type::data_type MapDataType(int32_t type) {
    switch (static_cast<OperandCode>(type)) {
        case ANEURALNETWORKS_FLOAT32:
        case ANEURALNETWORKS_TENSOR_FLOAT32:
            return slang::type::data_type::kFP32;
        case ANEURALNETWORKS_INT32:
        case ANEURALNETWORKS_TENSOR_INT32:
            return slang::type::data_type::kINT32;
        case ANEURALNETWORKS_UINT32:
            return slang::type::data_type::kUINT32;
        case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
            return slang::type::data_type::kUINT8;
        case ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL:
        case ANEURALNETWORKS_TENSOR_QUANT8_SYMM:
        case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED:
            return slang::type::data_type::kINT8;
        case ANEURALNETWORKS_BOOL:
        case ANEURALNETWORKS_TENSOR_BOOL8:
            return slang::type::data_type::kBOOL8;
        case ANEURALNETWORKS_TENSOR_QUANT16_ASYMM:
            return slang::type::data_type::kUINT16;
        case ANEURALNETWORKS_TENSOR_QUANT16_SYMM:
            return slang::type::data_type::kINT16;
        case ANEURALNETWORKS_TENSOR_FLOAT16:
        case ANEURALNETWORKS_FLOAT16:
            return slang::type::data_type::kFP16;
        default:
            std::cout << "Unknown data type from nnapi." << std::endl;
    }
    return slang::type::data_type::kINVALID;
}

slang::type::quant_type MapQuantType(int32_t type) {
    switch (static_cast<OperandCode>(type)) {
        case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
        case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED:
        case ANEURALNETWORKS_TENSOR_QUANT16_ASYMM:
            return slang::type::quant_type::kASYMM;
        case ANEURALNETWORKS_TENSOR_QUANT8_SYMM:
        case ANEURALNETWORKS_TENSOR_QUANT16_SYMM:
            return slang::type::quant_type::kSYMM;
        case ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL:
            return slang::type::quant_type::kSYMM_PCQ;
        default:
            return slang::type::quant_type::kNONE;
    }
    return slang::type::quant_type::kINVALID;
}

void PrintVXSpec(const tim::vx::TensorSpec& spec) {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Timvx tensor datatype: ";
    switch ((int32_t)spec.datatype_) {
        case 1:
            std::cout << "INT8" << std::endl;
            break;
        case 2:
            std::cout << "UINT8" << std::endl;
            break;
        case 3:
            std::cout << "INT16" << std::endl;
            break;
        case 4:
            std::cout << "UINT16" << std::endl;
            break;
        case 5:
            std::cout << "INT32" << std::endl;
            break;
        case 6:
            std::cout << "UINT32" << std::endl;
            break;
        case 7:
            std::cout << "INT64" << std::endl;
            break;
        case 8:
            std::cout << "FLOAT16" << std::endl;
            break;
        case 9:
            std::cout << "FLOAT32" << std::endl;
            break;
        case 10:
            std::cout << "BOOL8" << std::endl;
            break;
        default:
            std::cout << "Not support INT64 and other type";
            break;
    }
    std::cout << "Shape: ";
    for (auto it = spec.shape_.begin(); it != spec.shape_.end(); it++) {
        std::cout << *it << ",";
    }
    std::cout << std::endl;
    std::cout << "Attr: ";
    switch ((int32_t)spec.attr_) {
        case 1:
            std::cout << "CONSTANT" << std::endl;
            break;
        case 2:
            std::cout << "TRANSIENT" << std::endl;
            break;
        case 4:
            std::cout << "VARIABLE" << std::endl;
            break;
        case 8:
            std::cout << "INPUT" << std::endl;
            break;
        case 16:
            std::cout << "OUTPUT" << std::endl;
            break;
        default:
            std::cout << "Not support attr" << std::endl;
            break;
    }
    std::cout << "QuantType: ";
    switch ((int32_t)spec.quantization_.Type()) {
        case 0:
            std::cout << "NONE" << std::endl;
            break;
        case 1:
            std::cout << "ASYMMETRIC" << std::endl;
            break;
        case 2:
            std::cout << "SYMMETRIC_PER_CHANNEL" << std::endl;
            break;
        case 3:
            std::cout << "DYNAMIC_FIXED_POINT" << std::endl;
            break;
        case 16:
            std::cout << "OUTPUT" << std::endl;
            break;
        default:
            std::cout << "Not support quantization type" << std::endl;
            break;
    }
    if ((int32_t)spec.quantization_.Type() != 0) {
        std::cout << "Channel_dim: " << spec.quantization_.ChannelDim() << std::endl;
        std::cout << "Scales: ";
        PrintVector(spec.quantization_.Scales());
        std::cout << "Zero_points: ";
        PrintVector(spec.quantization_.ZeroPoints());
        std::cout << "Fl: " << spec.quantization_.Fl() << std::endl;
    }
}

void PrintTensorStorage(slang::type::tensor_storage s) {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Tensor storage datatype: ";
    switch ((int32_t)s.dtype) {
        case 0:
            std::cout << "kTF32" << std::endl;
            break;
        case 1:
            std::cout << "kFP32" << std::endl;
            break;
        case 2:
            std::cout << "kFP16" << std::endl;
            break;
        case 3:
            std::cout << "kBF16" << std::endl;
            break;
        case 4:
            std::cout << "kINT64" << std::endl;
            break;
        case 5:
            std::cout << "kINT32" << std::endl;
            break;
        case 6:
            std::cout << "kUINT32" << std::endl;
            break;
        case 7:
            std::cout << "kINT16" << std::endl;
            break;
        case 8:
            std::cout << "kUINT16" << std::endl;
            break;
        case 9:
            std::cout << "kINT8" << std::endl;
            break;
        case 10:
            std::cout << "kUINT8" << std::endl;
            break;
        case 11:
            std::cout << "kBOOL8" << std::endl;
            break;
        default:
            std::cout << "Not support tensor storage type" << std::endl;
            break;
    }
    std::cout << "data_length: " << s.data_length << std::endl;
    std::cout << "shape: ";  // original layout nhwc/nchw(not reverse)
    for (auto it = s.shape.begin(); it != s.shape.end(); it++) {
        std::cout << *it << ",";
    }
    std::cout << std::endl;
    std::cout << "attr: ";
    switch ((int32_t)s.attr) {
        case 0:
            std::cout << "kVARIABLE" << std::endl;
            break;
        case 1:
            std::cout << "kCONSTANT" << std::endl;
            break;
        default:
            std::cout << "Not support tensor torage attr" << std::endl;
            break;
    }
    std::cout << "quant_type: ";
    switch ((int32_t)s.qtype) {
        case 0:
            std::cout << "kNONE" << std::endl;
            break;
        case 1:
            std::cout << "kASYMM" << std::endl;
            break;
        case 2:
            std::cout << "kSYMM" << std::endl;
            break;
        case 3:
            std::cout << "kSYMM_PCQ" << std::endl;
            break;
        case 4:
            std::cout << "kDFP" << std::endl;
            break;
        default:
            std::cout << "Not support tensor torage qtype" << std::endl;
            break;
    }
    std::cout << "scale: " << s.scale << std::endl;
    std::cout << "zero_point: " << s.zero_point << std::endl;
    if (s.qtype == slang::type::quant_type::kSYMM_PCQ) {
        std::cout << "channel_dim: " << s.channel_dim << std::endl;
        std::cout << "per_channel_scales: ";
        PrintVector(s.per_channel_scales);
        std::cout << "per_channel_zero_points: ";
        PrintVector(s.per_channel_zero_points);
    }
}

void PrintScalarStorage(slang::type::scalar_storage s) {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "data_length: " << s.data.size() << std::endl;
    std::cout << "Scalar storage datatype: ";
    switch ((int32_t)s.dtype) {
        case 0:
            std::cout << "kTF32" << std::endl;
            break;
        case 1:
            std::cout << "kFP32" << std::endl;
            PrintScalarStorageData<float>(s);
            break;
        case 2:
            std::cout << "kFP16" << std::endl;
            break;
        case 3:
            std::cout << "kBF16" << std::endl;
            break;
        case 4:
            std::cout << "kINT64" << std::endl;
            PrintScalarStorageData<int64_t>(s);
            break;
        case 5:
            std::cout << "kINT32" << std::endl;
            PrintScalarStorageData<int32_t>(s);
            break;
        case 6:
            std::cout << "kUINT32" << std::endl;
            PrintScalarStorageData<uint32_t>(s);
            break;
        case 7:
            std::cout << "kINT16" << std::endl;
            PrintScalarStorageData<int16_t>(s);
            break;
        case 8:
            std::cout << "kUINT16" << std::endl;
            PrintScalarStorageData<uint16_t>(s);
            break;
        case 9:
            std::cout << "kINT8" << std::endl;
            PrintScalarStorageData<int8_t>(s);
            break;
        case 10:
            std::cout << "kUINT8" << std::endl;
            PrintScalarStorageData<uint8_t>(s);
            break;
        case 11:
            std::cout << "kBOOL8" << std::endl;
            PrintScalarStorageData<bool>(s);
            break;
        default:
            std::cout << "Not support scalar storage type" << std::endl;
            break;
    }
}
}  // namespace sl
}  // namespace android
}  // namespace vsi