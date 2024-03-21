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

#ifndef VSI_ANDROID_SL_MEMORY_DESC_H
#define VSI_ANDROID_SL_MEMORY_DESC_H

#include <android/NeuralNetworksTypes.h>

#include <optional>
#include <set>
#include <tuple>

#include "Types.h"
#include "slang/type_system.h"

namespace vsi::android::sl {

class Compilation;
using CompilationRole = std::tuple<const Compilation*, IOType, uint32_t>;

class MemoryDesc {
   public:
    int addRole(const Compilation* compilation, IOType ioType, uint32_t index, float frequency);
    int setShape(const std::vector<uint32_t>& dimensions);
    int finish();

    [[nodiscard]] bool finished() const { return finished_; }
    [[nodiscard]] size_t getSize() const;
    [[nodiscard]] std::set<CompilationRole> getRoles() const { return roles_; }
    [[nodiscard]] slang::type::tensor_storage getOperand() const { return *tensorOperand_; }
    [[nodiscard]] std::vector<uint32_t> getShape() const { return shape_; }

   private:
    std::set<CompilationRole> roles_;
    std::optional<slang::type::tensor_storage> tensorOperand_;
    std::vector<uint32_t> shape_;
    bool finished_ = false;
};

}  // namespace vsi::android::sl

#endif