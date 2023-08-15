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
#ifndef VSI_ANDROID_SL_VSI_DEVICE_H
#define VSI_ANDROID_SL_VSI_DEVICE_H

#include <cstdint>
#include <memory>
#include <string>

#include "tim/vx/platform/platform.h"

namespace vsi {
namespace android {
namespace sl {

class VsiDevice {
   public:
    VsiDevice(std::shared_ptr<tim::vx::platform::IDevice> device, std::string name)
        : device_(device), name_(name) {}
    const std::string& GetName() const { return name_; }
    const std::string& GetVersion() const { return version_; }
    const int64_t& GetFeatureLevel() const { return feature_level_; }
    std::shared_ptr<tim::vx::platform::IDevice> Device() const { return device_; }

   private:
    const std::string name_;
    const std::string version_{"0.0.1"};
    const int64_t feature_level_{1000006};
    std::shared_ptr<tim::vx::platform::IDevice> device_;
};

}  // namespace sl
}  // namespace android
}  // namespace vsi

#endif