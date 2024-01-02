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
#ifndef VSI_ANDROID_SL_UTILS_H_
#define VSI_ANDROID_SL_UTILS_H_
#include <android/NeuralNetworksTypes.h>
#include <android/log.h>

#include "Types.h"
#include "slang/type_system.h"
#include "tim/vx/types.h"

namespace vsi {
namespace android {
namespace sl {

#define LOG_TAG "NNAPI-VSI-SL"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL, LOG_TAG, __VA_ARGS__)

tim::vx::DataType ToTvxDataType(slang::type::data_type type);

tim::vx::QuantType ToTvxQuantType(slang::type::quant_type type);

slang::type::data_type MapDataType(int32_t type);

slang::type::quant_type MapQuantType(int32_t type);

void PrintVXSpec(const tim::vx::TensorSpec& spec);

void PrintTensorStorage(slang::type::tensor_storage s);

void PrintScalarStorage(slang::type::scalar_storage s);

template <typename T>
uint32_t GetTypeSize() {
    return sizeof(T);
}

template <typename T>
void PrintVector(const std::vector<T>& vector) {
    std::cout << "-------------------------------------------" << std::endl;
    for (auto it = vector.begin(); it != vector.end(); it++) {
        std::cout << *it << ",";
    }
    std::cout << std::endl;
}

template <typename T, unsigned long N>
void PrintArray(const std::array<T, N>& array) {
    std::cout << "-------------------------------------------" << std::endl;
    for (auto it = array.begin(); it != array.end(); it++) {
        std::cout << *it << ",";
    }
    std::cout << std::endl;
}

template <typename T>
void PrintScalarStorageData(slang::type::scalar_storage s) {
    std::cout << "scalar data: ";
    int length = s.data.size() / GetTypeSize<T>();
    for (int i = 0; i < length; ++i) {
        std::cout << *((T*)s.data.data() + i) << ",";
    }
    std::cout << "real length: " << length << std::endl;  // Real length of this type data
}

}  // namespace sl
}  // namespace android
}  // namespace vsi

#endif