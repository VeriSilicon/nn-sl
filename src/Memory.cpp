#include "Memory.h"

#include <android/hardware_buffer.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdlib>
#include <limits>

#include "Compilation.h"
#include "Utils.h"

namespace vsi::android::sl {

MemoryMapping::~MemoryMapping() {
    if (status_ != ANEURALNETWORKS_NO_ERROR) {
        return;
    }

    if (std::holds_alternative<int>(context_)) {
        int fd = std::get<int>(context_);
        if (fd > 0) {
            munmap(data_, size_);
        }
    } else if (std::holds_alternative<const AHardwareBuffer*>(context_)) {
        const auto* ahwb = std::get<const AHardwareBuffer*>(context_);
        AHardwareBuffer_unlock(const_cast<AHardwareBuffer*>(ahwb), nullptr);
    }
}

int IMemory::copy(const IMemory* src, const IMemory* dst) {
    if (src == dst) {
        return ANEURALNETWORKS_NO_ERROR;
    }

    if (!src->isInitialized()) {
        LOGE("IMemory::copy src memory is uninitialized");
        return ANEURALNETWORKS_BAD_DATA;
    }

    auto srcMapping = src->map();
    auto dstMapping = dst->map();

    if (srcMapping.getStatus() != ANEURALNETWORKS_NO_ERROR) {
        LOGE("IMemory::copy failed to map src memory");
        return srcMapping.getStatus();
    }

    if (dstMapping.getStatus() != ANEURALNETWORKS_NO_ERROR) {
        LOGE("IMemory::copy failed to map dst memory");
        return dstMapping.getStatus();
    }

    size_t srcSize = srcMapping.getSize();
    size_t dstSize = dstMapping.getSize();
    if (srcSize != dstSize) {
        LOGE("IMemory::copy src size (%zu) and dst size (%zu) not matched", srcSize, dstSize);
        return ANEURALNETWORKS_BAD_DATA;
    }

    const void* srcData = srcMapping.getData();
    void* dstData = dstMapping.getData();
    std::memcpy(dstData, srcData, srcSize);

    const_cast<IMemory*>(dst)->setInitialized(true);

    return ANEURALNETWORKS_NO_ERROR;
}

FdMemory* FdMemory::create(size_t size, int prot, int fd, size_t offset) {
    if (size == 0) {
        LOGE("FdMemory::create size is 0");
        return nullptr;
    }

    int memFd = dup(fd);
    if (memFd == -1) {
        LOGE("FdMemory::create failed to dup memory fd: %s (%d)", strerror(errno), errno);
        return nullptr;
    }

    auto* memory = new FdMemory(size, prot, memFd, offset);
    return memory;
}

FdMemory::~FdMemory() {
    close(fd_);
}

int FdMemory::validate(const Compilation* compilation, IOType ioType, uint32_t index,
                       const ANeuralNetworksOperandType* type, size_t offset, size_t length) const {
    if (offset + length > size_) {
        LOGE("FdMemory::validate requested size is larger than memory size");
        return ANEURALNETWORKS_BAD_DATA;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

MemoryMapping FdMemory::map() const {
    void* data = mmap(nullptr, size_, prot_, MAP_SHARED, fd_, static_cast<off_t>(offset_));
    if (data == MAP_FAILED) {
        LOGE("FdMemory::create failed to mmap fd: %s (%d)", strerror(errno), errno);
        return MemoryMapping(ANEURALNETWORKS_BAD_DATA);
    }

    return {data, size_, fd_};
}

AHardwareBufferMemory* AHardwareBufferMemory::create(const AHardwareBuffer* ahwb) {
    auto* memory = new AHardwareBufferMemory(ahwb);
    return memory;
}

size_t AHardwareBufferMemory::getSize() const {
    AHardwareBuffer_Desc desc;
    AHardwareBuffer_describe(ahwb_, &desc);
    return (desc.format == AHARDWAREBUFFER_FORMAT_BLOB) ? desc.width : 0;
}

int AHardwareBufferMemory::validate(const Compilation* compilation, IOType ioType, uint32_t index,
                                    const ANeuralNetworksOperandType* type, size_t offset,
                                    size_t length) const {
    AHardwareBuffer_Desc desc;
    AHardwareBuffer_describe(ahwb_, &desc);

    if (compilation == nullptr) {
        // The memory is used for constant tensors.

        if (desc.format != AHARDWAREBUFFER_FORMAT_BLOB) {
            LOGE("AHardwareBufferMemory::validate cannot set constant operand with non-blob "
                 "AHardwareBuffer memory");
            return ANEURALNETWORKS_BAD_DATA;
        }
    } else {
        // The memory is used for runtime I/O tensors.
        if (desc.format != AHARDWAREBUFFER_FORMAT_BLOB && (length != 0 || offset != 0)) {
            LOGE("AHardwareBufferMemory::validate both offset and length must be 0");
            return ANEURALNETWORKS_BAD_DATA;
        }

        if (offset + length > desc.width) {
            LOGE("DeviceMemory::validate requested size is larger than memory size");
            return ANEURALNETWORKS_BAD_DATA;
        }
    }
    return ANEURALNETWORKS_NO_ERROR;
}

MemoryMapping AHardwareBufferMemory::map() const {
    AHardwareBuffer_Desc desc;
    AHardwareBuffer_describe(ahwb_, &desc);

    if (desc.format != AHARDWAREBUFFER_FORMAT_BLOB) {
        LOGE("AHardwareBufferMemory::map unable to map non-blob AHardwareBuffer memory");
        return MemoryMapping(ANEURALNETWORKS_BAD_DATA);
    }
    uint32_t size = desc.width;

    constexpr uint64_t kCpuUsageMask =
            AHARDWAREBUFFER_USAGE_CPU_READ_MASK | AHARDWAREBUFFER_USAGE_CPU_WRITE_MASK;
    void* data = nullptr;
    int status = AHardwareBuffer_lock(const_cast<AHardwareBuffer*>(ahwb_),
                                      desc.usage & kCpuUsageMask, -1, nullptr, &data);
    if (status != 0) {
        LOGE("AHardwareBufferMemory::map cannot lock the AHardwareBuffer, error: %d", status);
        return MemoryMapping(ANEURALNETWORKS_BAD_DATA);
    }

    return {data, size, ahwb_};
}

DeviceMemory* DeviceMemory::create(const MemoryDesc* desc) {
    if (!desc->finished()) {
        LOGE("DeviceMemory::create cannot create device memory from an unfinished desc");
        return nullptr;
    }

    size_t size = desc->getSize();
    if (size == 0) {
        LOGE("DeviceMemory::create cannot create device memory from an zero-sized desc");
        return nullptr;
    }

    size_t alignedSize = alignSize(size, kAlignment);
    void* data = aligned_alloc(kAlignment, alignedSize);
    if (data == nullptr) {
        LOGE("DeviceMemory::create failed to allocate heap buffer");
        return nullptr;
    }

    auto* memory = new DeviceMemory(desc, data, size);
    return memory;
}

DeviceMemory::~DeviceMemory() {
    free(data_);
}

int DeviceMemory::validate(const Compilation* compilation, IOType ioType, uint32_t index,
                           const ANeuralNetworksOperandType* type, size_t offset,
                           size_t length) const {
    if (compilation == nullptr) {
        // The memory is used for constant tensors.
        LOGE("DeviceMemory::validate cannot set constant operand values with device memory");
        return ANEURALNETWORKS_BAD_DATA;
    }

    // The memory is used for runtime I/O tensors.
    if (length != 0 || offset != 0) {
        LOGE("DeviceMemory::validate both offset and length must be 0");
        return ANEURALNETWORKS_BAD_DATA;
    }

    if (roles_.count({compilation, ioType, index}) == 0) {
        LOGE("DeviceMemory::validate role not specified");
        return ANEURALNETWORKS_BAD_DATA;
    }

    const auto* model = compilation->getModel();
    const auto& tensorMap = model->getTensorMap();
    slang::type::tensor_storage tensorOperand;

    if (ioType == IOType::INPUT) {
        const auto& inputs = model->getInputs();
        if (index >= inputs.size()) {
            LOGE("MemoryDesc::validate input index (%u) out of range", index);
            return ANEURALNETWORKS_BAD_DATA;
        }

        uint32_t input = inputs[index];
        if (tensorMap.count(input) == 0) {
            LOGE("MemoryDesc::validate cannot find corresponding tensor for input index (%u)",
                 index);
            return ANEURALNETWORKS_BAD_DATA;
        }

        tensorOperand = tensorMap.at(input);
    } else if (ioType == IOType::OUTPUT) {
        const auto& outputs = model->getOutputs();
        if (index >= outputs.size()) {
            LOGE("MemoryDesc::validate output index (%u) out of range", index);
            return ANEURALNETWORKS_BAD_DATA;
        }

        uint32_t output = outputs[index];
        if (tensorMap.count(output) == 0) {
            LOGE("MemoryDesc::validate cannot find corresponding tensor for output index (%u)",
                 index);
            return ANEURALNETWORKS_BAD_DATA;
        }

        tensorOperand = tensorMap.at(output);
    }

    if (type != nullptr) {
        uint32_t rank = type->dimensionCount;
        if (shape_.size() != rank) {
            LOGE("DeviceMemory::validate incompatible tensor rank");
            return ANEURALNETWORKS_BAD_DATA;
        }

        for (size_t i = 0; i < rank; i++) {
            if (type->dimensions[i] == 0) {
                LOGE("DeviceMemory::validate dynamic (0-sized) axis is not supported");
                return ANEURALNETWORKS_OP_FAILED;
            }

            if (shape_[i] != type->dimensions[i]) {
                LOGE("DeviceMemory::validate incompatible dim length at axis %zu:"
                     " device memory (%u) vs. requested (%u)",
                     i, shape_[i], type->dimensions[i]);
                return ANEURALNETWORKS_BAD_DATA;
            }
        }

        if (tensorOperand.dtype != MapDataType(type->type) ||
            std::fabs(tensorOperand.scale - type->scale) > std::numeric_limits<float>::epsilon() ||
            tensorOperand.zero_point != type->zeroPoint) {
            LOGE("DeviceMemory::validate incompatible tensor metadata");
            return ANEURALNETWORKS_BAD_DATA;
        }
    }

    return ANEURALNETWORKS_NO_ERROR;
}

MemoryMapping DeviceMemory::map() const {
    return {data_, size_, nullptr};
}

}  // namespace vsi::android::sl