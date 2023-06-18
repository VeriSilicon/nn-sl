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
#ifndef VSI_ANDROID_SL_DEVICE_MANAGER_H
#define VSI_ANDROID_SL_DEVICE_MANAGER_H
#include <sys/system_properties.h>

#include <memory>
#include <cstdlib>

#include "VsiDevice.h"
#include "tim/vx/platform/native.h"
#ifdef USE_GRPC
#include "tim/vx/platform/grpc/grpc_remote.h"
#endif
namespace vsi {
namespace android {
namespace sl {
class DeviceManager {
   public:
    static DeviceManager* Instance() {
        if (instance_ == nullptr) {
            instance_ = new DeviceManager();
#ifdef USE_GRPC
            char env[32] = {0};
            __system_property_get("vendor.VSI_ASL_PORT", env);
            std::string port(env);
            auto devices = tim::vx::platform::GRPCRemoteDevice::Enumerate(port);

#else
            auto devices = tim::vx::platform::NativeDevice::Enumerate();
#endif
            for (int i = 0; i < devices.size(); ++i) {
                std::string name("vsi-device-" + std::to_string(i));
                std::shared_ptr<VsiDevice> device = std::make_shared<VsiDevice>(devices[i], name);
                instance_->GetDevices().push_back(device);
            }
        }
        return instance_;
    }
    std::vector<std::shared_ptr<VsiDevice>>& GetDevices() { return devices_; }

   private:
    DeviceManager(){};
    DeviceManager(const DeviceManager&){};
    DeviceManager& operator=(const DeviceManager&) = delete;

    static DeviceManager* instance_;
    std::vector<std::shared_ptr<VsiDevice>> devices_;
};

DeviceManager* DeviceManager::instance_ = nullptr;

}  // namespace sl
}  // namespace android
}  // namespace vsi

#endif