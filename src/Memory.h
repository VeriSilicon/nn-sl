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

#ifndef VSI_ANDROID_SL_MEMORY_H
#define VSI_ANDROID_SL_MEMORY_H

#include <android/NeuralNetworksTypes.h>
#include <android/hardware_buffer.h>

#include <utility>
#include <variant>

#include "MemoryDesc.h"
#include "Types.h"

namespace vsi::android::sl {

struct MemoryMapping {
   public:
    using Context = std::variant<int, const AHardwareBuffer*, nullptr_t>;

    explicit MemoryMapping(int status) : status_(status), data_(nullptr), size_(0) {}
    MemoryMapping(void* data, size_t size, Context context)
        : status_(ANEURALNETWORKS_NO_ERROR), data_(data), size_(size), context_(context) {}

    MemoryMapping(const MemoryMapping&) = delete;
    MemoryMapping& operator=(const MemoryMapping&) = delete;

    ~MemoryMapping();

    [[nodiscard]] int getStatus() const { return status_; }
    [[nodiscard]] void* getData() const { return data_; }
    [[nodiscard]] size_t getSize() const { return size_; }

   private:
    int status_;
    void* data_;
    size_t size_;
    Context context_;
};

class IMemory {
   public:
    virtual ~IMemory() = default;

    [[nodiscard]] virtual size_t getSize() const = 0;
    [[nodiscard]] virtual int validate(const Compilation* compilation, IOType ioType,
                                       uint32_t index, const ANeuralNetworksOperandType* type,
                                       size_t offset, size_t length) const = 0;
    [[nodiscard]] virtual MemoryMapping map() const = 0;

    [[nodiscard]] virtual bool isInitialized() const = 0;
    virtual void setInitialized(bool initialized) = 0;

    static int copy(const IMemory* src, const IMemory* dst);
};

class FdMemory final : public IMemory {
   public:
    explicit FdMemory(size_t size, int prot, int fd, size_t offset)
        : size_(size), prot_(prot), fd_(fd), offset_(offset) {}
    ~FdMemory() override;
    static FdMemory* create(size_t size, int prot, int fd, size_t offset);

    [[nodiscard]] size_t getSize() const override { return size_; }
    [[nodiscard]] int validate(const Compilation* compilation, IOType ioType, uint32_t index,
                               const ANeuralNetworksOperandType* type, size_t offset,
                               size_t length) const override;
    [[nodiscard]] MemoryMapping map() const override;
    [[nodiscard]] bool isInitialized() const override { return fd_ != -1; }
    void setInitialized(bool initialized) override {}

   private:
    int fd_ = -1;
    int prot_ = 0;
    size_t size_ = 0;
    size_t offset_ = 0;
};

class AHardwareBufferMemory final : public IMemory {
   public:
    explicit AHardwareBufferMemory(const AHardwareBuffer* ahwb) : ahwb_(ahwb) {}
    ~AHardwareBufferMemory() override = default;
    static AHardwareBufferMemory* create(const AHardwareBuffer* ahwb);

    [[nodiscard]] size_t getSize() const override;
    [[nodiscard]] int validate(const Compilation* compilation, IOType ioType, uint32_t index,
                               const ANeuralNetworksOperandType* type, size_t offset,
                               size_t length) const override;
    [[nodiscard]] MemoryMapping map() const override;
    [[nodiscard]] bool isInitialized() const override { return ahwb_ != nullptr; }
    void setInitialized(bool initialized) override {}

   private:
    const AHardwareBuffer* ahwb_;
};

class DeviceMemory final : public IMemory {
    static constexpr size_t kAlignment = 64;

   public:
    explicit DeviceMemory(const MemoryDesc* desc, void* data, size_t size)
        : roles_(desc->getRoles()),
          tensorOperand_(desc->getOperand()),
          shape_(desc->getShape()),
          data_(data),
          size_(size) {}
    ~DeviceMemory() override;
    static DeviceMemory* create(const MemoryDesc* desc);

    [[nodiscard]] size_t getSize() const override { return size_; }
    [[nodiscard]] int validate(const Compilation* compilation, IOType ioType, uint32_t index,
                               const ANeuralNetworksOperandType* type, size_t offset,
                               size_t length) const override;
    [[nodiscard]] MemoryMapping map() const override;
    [[nodiscard]] bool isInitialized() const override { return initialized_; }
    void setInitialized(bool initialized) override { initialized_ = initialized; }

   private:
    std::set<CompilationRole> roles_;
    slang::type::tensor_storage tensorOperand_;
    std::vector<uint32_t> shape_;

    void* data_;
    size_t size_;
    bool initialized_ = false;
};

}  // namespace vsi::android::sl

#endif