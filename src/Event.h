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

#include "NeuralNetworksTypes.h"
#include "Types.h"
#include "Execution.h"

namespace vsi {
namespace android {
namespace sl {
class Event {
   public:
    Event() {}
    Event(int sync_fence) : sync_fence_(sync_fence) {}
    Event(Execution exec, Event eve) : execution_(exec) {}

   private:
    int sync_fence_{0};
    Execution execution_{nullptr};
    Event depend_{nullptr};
    bool finished_{false};
    void* data_{nullptr};
    size_t length_{0};
};

}  // namespace sl
}  // namespace android
}  // namespace vsi
#endif