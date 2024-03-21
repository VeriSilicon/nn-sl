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

#ifndef VSI_ANDROID_SL_EVENT_H
#define VSI_ANDROID_SL_EVENT_H

#include <condition_variable>
#include <mutex>
#include <thread>

#include "Types.h"

namespace vsi::android::sl {

class IEvent {
   public:
    virtual ~IEvent() = default;
    virtual int wait() const = 0;  // NOLINT(modernize-use-nodiscard)
    [[nodiscard]] virtual int getSyncFenceFd(bool shouldDup) const = 0;
};

class CallbackEvent : public IEvent {
   public:
    explicit CallbackEvent(TimePoint deadline);

    [[nodiscard]] int getSyncFenceFd(bool /*shouldDup*/) const override { return -1; }

    int bindThread(std::thread thread);
    int wait() const override;
    void notify();

   private:
    mutable std::thread thread_;
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    bool isNotified_;
    TimePoint deadline_;
};

class SyncFenceEvent : public IEvent {
   public:
    explicit SyncFenceEvent(int syncFenceFd);
    ~SyncFenceEvent() override;

    [[nodiscard]] int getSyncFenceFd(bool shouldDup) const override;

    int wait() const override;

   private:
    int syncFenceFd_ = -1;
    mutable std::mutex mutex_;
};

}  // namespace vsi::android::sl
#endif