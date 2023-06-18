/****************************************************************************
 *
 *    Copyright (c) 2023 Vivante Corporation
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

#include "NeuralNetworksTypes.h"
#include "Types.h"

namespace vsi {
namespace android {
namespace sl {
class Memory {
   public:
    ~Memory() {
        if (create_from_fd_) munmap(data_, length_);
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
    void* Data() const { return data_; }
    void* Data() { return data_; }
    const size_t Length() const { return length_; }
    size_t Length() { return length_; }

   private:
    bool create_from_fd_{false};
    void* data_{nullptr};
    size_t length_{0};
};

}  // namespace sl
}  // namespace android
}  // namespace vsi
#endif