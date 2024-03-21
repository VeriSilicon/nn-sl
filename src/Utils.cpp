/****************************************************************************
 *
 *    Copyright (c) 2022 Vivante Corporation
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

#include "Utils.h"

namespace vsi::android::sl {

size_t getDtypeSize(slang::type::data_type type) {
    // NOLINTBEGIN(*-magic-numbers)
    switch (type) {
        case slang::type::data_type::kINT8:
        case slang::type::data_type::kUINT8:
        case slang::type::data_type::kBOOL8:
            return 1;
        case slang::type::data_type::kINT16:
        case slang::type::data_type::kUINT16:
        case slang::type::data_type::kFP16:
        case slang::type::data_type::kBF16:
            return 2;
        case slang::type::data_type::kINT32:
        case slang::type::data_type::kUINT32:
        case slang::type::data_type::kFP32:
            return 4;
        case slang::type::data_type::kINT64:
            return 8;
        default:
            return 0;
    }
    // NOLINTEND(*-magic-numbers)
}

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
            LOGW("Unsupported slang dtype: %d", type);
            return tim::vx::DataType::UNKNOWN;
    }
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
        case ANEURALNETWORKS_MODEL:
            return slang::type::data_type::kMODELVALUE;
        default:
            LOGW("Unsupported NNAPI dtype: %d", type);
            return slang::type::data_type::kINVALID;
    }
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

Shape combineShape(const Shape& lhs, const Shape& rhs) {
    if (rhs.empty()) {
        return lhs;
    }

    if (lhs.empty()) {
        return rhs;
    }

    if (lhs.size() != rhs.size()) {
        LOGE("%s incompatible ranks: lhs (%zu) vs. rhs (%zu)", __func__, lhs.size(), rhs.size());
        return {};
    }

    Shape combined = lhs;
    for (size_t i = 0; i < lhs.size(); i++) {
        if (lhs[i] == 0) {
            combined[i] = rhs[i];
        } else if (rhs[i] != 0 && lhs[i] != rhs[i]) {
            LOGE("%s incompatible dim length at axis %zu: lhs (%u) vs. rhs (%u)", __func__, i,
                 lhs[i], rhs[i]);
            return {};
        }
    }

    return combined;
}

}  // namespace vsi::android::sl