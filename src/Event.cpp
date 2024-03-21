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

#include "Event.h"

#include <sys/fcntl.h>
#include <sys/poll.h>
#include <sys/signal.h>
#include <unistd.h>

#include <condition_variable>
#include <mutex>
#include <thread>

#include "Utils.h"

namespace vsi::android::sl {

CallbackEvent::CallbackEvent(TimePoint deadline) {
    deadline_ = deadline;
    isNotified_ = false;
}

int CallbackEvent::bindThread(std::thread thread) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (thread_.joinable()) {
        LOGE("CallbackEvent::bindThread a thread is already bound");
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (!thread.joinable()) {
        LOGE("CallbackEvent::bindThread passed an invalid thread");
        return ANEURALNETWORKS_BAD_STATE;
    }

    thread_ = std::move(thread);
    return ANEURALNETWORKS_NO_ERROR;
}

int CallbackEvent::wait() const {
    std::unique_lock<std::mutex> lock(mutex_);

    if (!cv_.wait_until(lock, deadline_, [this] { return isNotified_; })) {
        return ANEURALNETWORKS_MISSED_DEADLINE_TRANSIENT;
    }

    if (thread_.joinable()) {
        thread_.join();
    }

    return ANEURALNETWORKS_NO_ERROR;
}

void CallbackEvent::notify() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (isNotified_) {
            return;
        }
        isNotified_ = true;
    }
    cv_.notify_all();
}

SyncFenceEvent::SyncFenceEvent(int syncFenceFd) {
    if (syncFenceFd > 0) {
        syncFenceFd_ = dup(syncFenceFd);
    }
}

SyncFenceEvent::~SyncFenceEvent() {
    close(syncFenceFd_);
}

int SyncFenceEvent::getSyncFenceFd(bool shouldDup) const {
    int syncFenceFd = shouldDup ? dup(syncFenceFd_) : syncFenceFd_;
    return syncFenceFd;
}

int SyncFenceEvent::wait() const {
    if (syncFenceFd_ == -1) {
        // The SL don't support creating sync fence.
        return ANEURALNETWORKS_NO_ERROR;
    }
    // if (syncFenceFd_ < 0) {
    //     errno = EINVAL;
    //     return ANEURALNETWORKS_BAD_STATE;
    // }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        struct pollfd fds;
        fds.fd = syncFenceFd_;
        fds.events = POLLIN;
        int timeout = -1;

        int ret;
        do {
            ret = poll(&fds, 1, timeout);
            if (ret > 0) {
                if ((fds.revents & POLLNVAL) != 0) {
                    errno = EINVAL;
                    return ANEURALNETWORKS_BAD_STATE;
                }
                if ((fds.revents & POLLERR) != 0) {
                    errno = EINVAL;
                    return ANEURALNETWORKS_BAD_STATE;
                }
                // Signaled.
                return ANEURALNETWORKS_NO_ERROR;
            }
            if (ret == 0) {
                // Timeouted.
                errno = ETIME;
                return ANEURALNETWORKS_MISSED_DEADLINE_TRANSIENT;
            }
        } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
    }

    return ANEURALNETWORKS_BAD_STATE;
}

}  // namespace vsi::android::sl