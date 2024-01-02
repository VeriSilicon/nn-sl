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
#ifndef VSI_ANDROID_SL_MEMORY_TYPE_H
#define VSI_ANDROID_SL_MEMORY_TYPE_H

#include <sys/mman.h>
#include <android/hardware_buffer.h>
#include <dlfcn.h>

#include <android/NeuralNetworksTypes.h>
#include "Types.h"

namespace vsi {
namespace android {
namespace sl {

class MemoryDesc {
   public:
    void UpdateDataSize() {
        auto dims = 1;
        for (int i = 0; i < shape_.size(); ++i) dims *= shape_.at(i);
        switch (t_storage_.dtype) {
            case slang::type::data_type::kINT64:
                length_ = dims * 8;
                break;
            case slang::type::data_type::kFP32:
            case slang::type::data_type::kTF32:
            case slang::type::data_type::kINT32:
            case slang::type::data_type::kUINT32:
                length_ = dims * 4;
                break;
            case slang::type::data_type::kFP16:
            case slang::type::data_type::kBF16:
            case slang::type::data_type::kINT16:
            case slang::type::data_type::kUINT16:
                length_ = dims * 2;
                break;
            case slang::type::data_type::kUINT8:
            case slang::type::data_type::kINT8:
            case slang::type::data_type::kBOOL8:
                length_ = dims;
                break;
            case slang::type::data_type::kUINT4:
            case slang::type::data_type::kINT4:
                length_ = dims / 2;
                break;
            default:
                std::cout << "Invalid data type corresponding to the role" << std::endl;
                break;
        }
        if (length_ <= 0) std::cout << "Invalid shape corresponding to the desc" << std::endl;
    }
    int SetDimensions(const std::vector<uint32_t>& shape) {
        shape_ = shape;
        return ANEURALNETWORKS_NO_ERROR;
    }
    int AddRole(TensorMap& tensor_map, IOType io_type, uint32_t operand_id, float freq) {
        // To do: check index validity
        t_storage_ = tensor_map.at(operand_id);
        io_type_ = io_type;
        UpdateDataSize();
        return ANEURALNETWORKS_NO_ERROR;
    }
    int Finish() {
        finished_ = true;
        return ANEURALNETWORKS_NO_ERROR;
    }
    const size_t Length() const { return length_; }
    size_t Length() { return length_; }
    std::vector<uint32_t> Shape() const { return shape_; }
    std::vector<uint32_t>& Shape() { return shape_; }
    bool IsFinished() const { return finished_; }

   private:
    IOType io_type_;
    std::vector<uint32_t> shape_;
    slang::type::tensor_storage t_storage_;
    bool finished_{false};
    size_t length_{0};
};
class Memory {
   public:
    ~Memory() {
        if (create_from_fd_) munmap(data_, length_);
        if (create_from_ahwb_ || create_from_desc_) free(data_);
    }
    int CreateFromFd(size_t size, int prot, int fd, size_t offset) {
        if (size <= 0) {
            std::cout << "Invalid size" << std::endl;
            return ANEURALNETWORKS_BAD_DATA;
        }
        data_ = mmap(nullptr, size, prot, MAP_SHARED, fd, offset);
        if (data_ == MAP_FAILED) {
            std::cout << "Can't mmap with the fd." << std::endl;
            return ANEURALNETWORKS_BAD_STATE;
        }
        length_ = size;
        create_from_fd_ = true;

        return ANEURALNETWORKS_NO_ERROR;
    }
    int CreateFromAHWB(const AHardwareBuffer* ahwb) {
        if (ahwb == nullptr) {
            std::cout << "Invalid AHardwareBuffer pointer" << std::endl;
            return ANEURALNETWORKS_BAD_DATA;
        }
        ahwb_ = ahwb;
        create_from_ahwb_ = true;

        return ANEURALNETWORKS_NO_ERROR;
    }
    int PraseAHWB(const AHardwareBuffer* ahwb) {
        AHardwareBuffer_Desc desc = {0};
        AHardwareBuffer_describe(ahwb, &desc);
        if (desc.format != AHARDWAREBUFFER_FORMAT_BLOB) {
            std::cout << "Unable to map non-blob AHardwareBuffer memory" << std::endl;
            return ANEURALNETWORKS_BAD_DATA;
        }
        const uint32_t size = desc.width;
        void* buffer = (void*)malloc(size);
        if (buffer == nullptr) {
            std::cout << "Malloc buffer fail" << std::endl;
            return ANEURALNETWORKS_BAD_STATE;
        }
        const uint64_t kCpuUsageMask =
                AHARDWAREBUFFER_USAGE_CPU_READ_MASK | AHARDWAREBUFFER_USAGE_CPU_WRITE_MASK;
        void* data = nullptr;
        auto status = AHardwareBuffer_lock(const_cast<AHardwareBuffer*>(ahwb),
                                           desc.usage & kCpuUsageMask, -1, nullptr, &data);
        if (status != /*NO_ERROR*/ 0) {
            std::cout << "HardwareBuffer lock memory fail" << std::endl;
            return ANEURALNETWORKS_BAD_DATA;
        }
        memcpy((void*)buffer, (void*)data, size);
        data_ = buffer;
        length_ = size;
        status = AHardwareBuffer_unlock(const_cast<AHardwareBuffer*>(ahwb), nullptr);
        if (status != /*NO_ERROR*/ 0) {
            std::cout << "HardwareBuffer unlock memory fail" << std::endl;
            return ANEURALNETWORKS_BAD_DATA;
        }
        return ANEURALNETWORKS_NO_ERROR;
    }
    int CreateFromDesc(const MemoryDesc* mdesc) {
        create_from_desc_ = true;
        mdesc_ = mdesc;
        size_t length = mdesc->Length();
        void* buffer = (void*)malloc(length);
        if (buffer == nullptr) {
            std::cout << "Malloc buffer fail" << std::endl;
            return ANEURALNETWORKS_BAD_STATE;
        }
        data_ = buffer;
        length_ = length;
        return ANEURALNETWORKS_NO_ERROR;
    }
    const MemoryDesc* GetDesc() { return mdesc_; }
    void* Data() const { return data_; }
    void* Data() { return data_; }
    void SetData(void* buffer) { data_ = buffer; }
    const size_t Length() const { return length_; }
    size_t Length() { return length_; }
    void SetLength(size_t length) { length_ = length; }
    bool IsCreateFromFd() const { return create_from_fd_; }
    bool IsCreateFromAHWB() const { return create_from_ahwb_; }
    bool IsCreateFromDesc() const { return create_from_desc_; }
    const AHardwareBuffer* AHWB() const { return ahwb_; }

   private:
    bool create_from_fd_{false};
    bool create_from_ahwb_{false};
    bool create_from_desc_{false};
    const AHardwareBuffer* ahwb_{nullptr};
    const MemoryDesc* mdesc_{nullptr};
    void* data_{nullptr};
    size_t length_{0};
};

}  // namespace sl
}  // namespace android
}  // namespace vsi
#endif