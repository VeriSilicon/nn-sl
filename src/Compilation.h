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

#ifndef VSI_ANDROID_SL_COMPILATION_H
#define VSI_ANDROID_SL_COMPILATION_H

#include <array>
#include <filesystem>
#include <memory>
#include <vector>

#include "Device.h"
#include "Model.h"
#include "Types.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"

namespace vsi::android::sl {

namespace fs = std::filesystem;

class Compilation {
   public:
    enum class CacheState {
        DISABLED,
        EMPTY,
        LOADED,
    };

    static constexpr uint32_t kNumModelCacheFiles = 1;
    static constexpr uint32_t kNumDataCacheFiles = 0;

    explicit Compilation(Model* model)
        : model_(model),
          cacheState_(CacheState::DISABLED),
          vxContext_(tim::vx::Context::Create()) {}
    explicit Compilation(Model* model, const std::vector<std::shared_ptr<Device>>& devices)
        : model_(model),
          devices_(devices),
          cacheState_(CacheState::DISABLED),
          vxContext_(tim::vx::Context::Create()) {}

    ~Compilation();
    int finish();
    [[nodiscard]] bool isFinished() const { return finished_; }
    [[nodiscard]] bool isBurst() const { return isBurst_; }

    [[nodiscard]] Model* getModel() { return model_; }
    [[nodiscard]] const Model* getModel() const { return model_; }
    [[nodiscard]] const std::vector<std::shared_ptr<Device>>& getDevices() const {
        return devices_;
    }

    [[nodiscard]] CacheState getCacheState() const { return cacheState_; }
    [[nodiscard]] const uint8_t* getCacheData() const {
        return cacheBuffer_.empty() ? nullptr : cacheBuffer_.data();
    }
    [[nodiscard]] size_t getCacheSize() const { return cacheBuffer_.size(); }
    int writeToCache(const uint8_t* data, size_t size);

    int setPreference(PreferenceCode preference);
    int setPriority(PriorityCode priority);
    int setTimeout(Duration duration);
    int setCaching(int fd, const uint8_t* token);
    int setCaching(const fs::path& cacheDir, const uint8_t* token);

    void setBurst() { isBurst_ = true; }
    void setCompiledGraph(const std::shared_ptr<tim::vx::Graph>& compiledGraph) {
        vxGraph_ = compiledGraph;
    }
    [[nodiscard]] std::shared_ptr<tim::vx::Context> getContext() { return vxContext_; }
    [[nodiscard]] std::shared_ptr<tim::vx::Graph> getCompiledGraph() { return vxGraph_; }

   private:
    static constexpr std::array<char, 4> kNBGMagic = {'V', 'P', 'M', 'N'};

    Model* model_;
    PreferenceCode preference_;
    PriorityCode priority_;
    Duration timeoutDuration_;
    std::vector<std::shared_ptr<Device>> devices_;
    std::shared_ptr<tim::vx::Context> vxContext_;
    std::shared_ptr<tim::vx::Graph> vxGraph_;

    CacheState cacheState_;
    std::vector<uint8_t> cacheBuffer_;
    std::array<uint8_t, ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN> cacheToken_;
    int cacheFd_ = -1;

    bool finished_ = false;
    bool isBurst_ = false;
};

}  // namespace vsi::android::sl

#endif