/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/****************************************************************************
 *
 *    Copyright (c) 2024 Vivante Corporation
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software" =
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

#include <android/NeuralNetworks.h>

#include <memory>
#include <utility>
#include <vector>

#include "Compilation.h"
#include "Device.h"
#include "DeviceManager.h"
#include "Event.h"
#include "Execution.h"
#include "Memory.h"
#include "MemoryDesc.h"
#include "Model.h"
#include "NeuralNetworksSupportLibraryImpl.h"
#include "Utils.h"

using namespace vsi::android::sl;

int ANeuralNetworks_getDeviceCount(uint32_t* numDevices) {
    LOGV(__func__);
    if (numDevices == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    *numDevices = DeviceManager::get()->getNumDevices();
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworks_getDevice(uint32_t devIndex, ANeuralNetworksDevice** device) {
    LOGV(__func__);
    if (device == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    const auto& devices = DeviceManager::get()->getDevices();
    if (devIndex >= devices.size()) {
        LOGE("%s passed an invalid device index", __func__);
        return ANEURALNETWORKS_BAD_DATA;
    }

    *device = reinterpret_cast<ANeuralNetworksDevice*>(devices[devIndex].get());
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksDevice_getName(const ANeuralNetworksDevice* device, const char** name) {
    LOGV(__func__);
    if (device == nullptr || name == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    const auto* dev = reinterpret_cast<const Device*>(device);
    *name = dev->getName().data();
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksDevice_getVersion(const ANeuralNetworksDevice* device, const char** version) {
    LOGV(__func__);
    if (device == nullptr || version == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    const auto* dev = reinterpret_cast<const Device*>(device);
    *version = dev->getVersion().data();
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksDevice_getType(const ANeuralNetworksDevice* device, int32_t* type) {
    LOGV(__func__);
    if (device == nullptr || type == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    *type = ANEURALNETWORKS_DEVICE_ACCELERATOR;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksDevice_getFeatureLevel(const ANeuralNetworksDevice* device,
                                          int64_t* featureLevel) {
    LOGV(__func__);
    if (device == nullptr || featureLevel == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    const auto* dev = reinterpret_cast<const Device*>(device);
    *featureLevel = dev->getFeatureLevel();
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksDevice_wait(const ANeuralNetworksDevice* device) {
    LOGV(__func__);
    if (device == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_getSupportedOperationsForDevices(
        const ANeuralNetworksModel* model, const ANeuralNetworksDevice* const* devices,
        uint32_t numDevices, bool* supportedOps) {
    LOGV(__func__);

    if (model == nullptr || devices == nullptr || supportedOps == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    if (numDevices == 0) {
        LOGE("%s passed an empty device list", __func__);
        return ANEURALNETWORKS_BAD_DATA;
    }

    const auto* m = reinterpret_cast<const Model*>(model);
    return m->getSupportedOperations(supportedOps);
}

int ANeuralNetworksBurst_create(ANeuralNetworksCompilation* compilation,
                                ANeuralNetworksBurst** burst) {
    LOGV(__func__);

    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksBurst_free(ANeuralNetworksBurst* burst) {
    LOGV(__func__);
}

int ANeuralNetworksExecution_burstCompute(ANeuralNetworksExecution* execution,
                                          ANeuralNetworksBurst* burst) {
    LOGV(__func__);

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksMemoryDesc_create(ANeuralNetworksMemoryDesc** desc) {
    LOGV(__func__);
    if (desc == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* memDesc = new MemoryDesc();
    *desc = reinterpret_cast<ANeuralNetworksMemoryDesc*>(memDesc);
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksMemoryDesc_free(ANeuralNetworksMemoryDesc* desc) {
    LOGV(__func__);
    if (desc == nullptr) {
        LOGD("%s passed a nullptr", __func__);
        return;
    }

    auto* memDesc = reinterpret_cast<MemoryDesc*>(desc);
    delete memDesc;
}

int ANeuralNetworksMemoryDesc_addInputRole(ANeuralNetworksMemoryDesc* desc,
                                           const ANeuralNetworksCompilation* compilation,
                                           uint32_t index, float frequency) {
    LOGV(__func__);
    if (desc == nullptr || compilation == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* memDesc = reinterpret_cast<MemoryDesc*>(desc);
    const auto* c = reinterpret_cast<const Compilation*>(compilation);
    return memDesc->addRole(c, IOType::INPUT, index, frequency);
}

int ANeuralNetworksMemoryDesc_addOutputRole(ANeuralNetworksMemoryDesc* desc,
                                            const ANeuralNetworksCompilation* compilation,
                                            uint32_t index, float frequency) {
    LOGV(__func__);
    if (desc == nullptr || compilation == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* memDesc = reinterpret_cast<MemoryDesc*>(desc);
    const auto* c = reinterpret_cast<const Compilation*>(compilation);
    return memDesc->addRole(c, IOType::OUTPUT, index, frequency);
}

int ANeuralNetworksMemoryDesc_setDimensions(ANeuralNetworksMemoryDesc* desc, uint32_t rank,
                                            const uint32_t* dimensions) {
    LOGV(__func__);

    if (desc == nullptr || (dimensions == nullptr && rank > 0)) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* memDesc = reinterpret_cast<MemoryDesc*>(desc);
    const std::vector<uint32_t> shape(dimensions, dimensions + rank);
    return memDesc->setShape(shape);
}

int ANeuralNetworksMemoryDesc_finish(ANeuralNetworksMemoryDesc* desc) {
    LOGV(__func__);
    if (desc == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* memDesc = reinterpret_cast<MemoryDesc*>(desc);
    return memDesc->finish();
}

int ANeuralNetworksMemory_createFromDesc(const ANeuralNetworksMemoryDesc* desc,
                                         ANeuralNetworksMemory** memory) {
    LOGV(__func__);
    if (desc == nullptr || memory == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    const auto* memDesc = reinterpret_cast<const MemoryDesc*>(desc);
    auto* mem = DeviceMemory::create(memDesc);
    if (mem == nullptr) {
        LOGE("%s failed to create device memory from desc", __func__);
        return ANEURALNETWORKS_OP_FAILED;
    }

    *memory = reinterpret_cast<ANeuralNetworksMemory*>(mem);
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksMemory_createFromFd(size_t size, int prot, int fd, size_t offset,
                                       ANeuralNetworksMemory** memory) {
    LOGV(__func__);

    if (memory == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* mem = FdMemory::create(size, prot, fd, offset);
    if (mem == nullptr) {
        LOGE("%s failed to create memory from fd (%d)", __func__, fd);
        return ANEURALNETWORKS_BAD_DATA;
    }

    *memory = reinterpret_cast<ANeuralNetworksMemory*>(mem);
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksMemory_createFromAHardwareBuffer(const AHardwareBuffer* ahwb,
                                                    ANeuralNetworksMemory** memory) {
    LOGV(__func__);

    if (ahwb == nullptr || memory == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* mem = AHardwareBufferMemory::create(ahwb);
    if (mem == nullptr) {
        LOGE("%s failed to create memory from ahwb", __func__);
        return ANEURALNETWORKS_BAD_DATA;
    }

    *memory = reinterpret_cast<ANeuralNetworksMemory*>(mem);
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory) {
    LOGV(__func__);
    if (memory == nullptr) {
        LOGD("%s passed a nullptr", __func__);
        return;
    }

    auto* mem = reinterpret_cast<IMemory*>(memory);
    delete mem;
}

int ANeuralNetworksMemory_copy(const ANeuralNetworksMemory* src, const ANeuralNetworksMemory* dst) {
    LOGV(__func__);
    if (src == nullptr || dst == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    const auto* srcMem = reinterpret_cast<const IMemory*>(src);
    const auto* dstMem = reinterpret_cast<const IMemory*>(dst);
    return IMemory::copy(srcMem, dstMem);
}

int ANeuralNetworksModel_create(ANeuralNetworksModel** model) {
    LOGV(__func__);

    if (model == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* m = new Model();
    *model = reinterpret_cast<ANeuralNetworksModel*>(m);
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksModel_free(ANeuralNetworksModel* model) {
    LOGV(__func__);

    if (model == nullptr) {
        LOGV("%s passed a nullptr", __func__);
        return;
    }

    auto* m = reinterpret_cast<Model*>(model);
    delete m;
}

int ANeuralNetworksModel_finish(ANeuralNetworksModel* model) {
    LOGV(__func__);

    if (model == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* m = reinterpret_cast<Model*>(model);
    return m->finish();
}

int ANeuralNetworksModel_addOperand(ANeuralNetworksModel* model,
                                    const ANeuralNetworksOperandType* type) {
    LOGV(__func__);

    if (model == nullptr || type == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* m = reinterpret_cast<Model*>(model);
    return m->addOperand(*type);
}

int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model, int32_t index,
                                         const void* buffer, size_t length) {
    LOGV(__func__);

    if (model == nullptr || (buffer == nullptr && length != 0)) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* m = reinterpret_cast<Model*>(model);
    return m->setOperandValue(index, buffer, length);
}

int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel* model, int32_t index,
                                                   const ANeuralNetworksMemory* memory,
                                                   size_t offset, size_t length) {
    LOGV(__func__);

    if (model == nullptr || memory == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* m = reinterpret_cast<Model*>(model);
    const auto* mem = reinterpret_cast<const IMemory*>(memory);
    return m->setOperandValueFromMemory(index, mem, offset, length);
}

int ANeuralNetworksModel_setOperandValueFromModel(ANeuralNetworksModel* model, int32_t index,
                                                  const ANeuralNetworksModel* value) {
    LOGV(__func__);

    if (model == nullptr || value == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* m = reinterpret_cast<Model*>(model);
    const auto* reference = reinterpret_cast<const Model*>(value);
    return m->setOperandValueFromModel(index, reference);
}

int ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                      ANeuralNetworksOperationType type, uint32_t inputCount,
                                      const uint32_t* inputs, uint32_t outputCount,
                                      const uint32_t* outputs) {
    LOGV(__func__);

    if (model == nullptr || inputs == nullptr || outputs == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* m = reinterpret_cast<Model*>(model);
    return m->addOperation(type, inputCount, inputs, outputCount, outputs);
}

int ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(
        ANeuralNetworksModel* model, int32_t index,
        const ANeuralNetworksSymmPerChannelQuantParams* channelQuant) {
    LOGV(__func__);

    if (model == nullptr || channelQuant == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* m = reinterpret_cast<Model*>(model);
    return m->setOperandSymmPerChannelQuantParams(index, *channelQuant);
}

int ANeuralNetworksModel_identifyInputsAndOutputs(ANeuralNetworksModel* model, uint32_t inputCount,
                                                  const uint32_t* inputs, uint32_t outputCount,
                                                  const uint32_t* outputs) {
    LOGV(__func__);

    if (model == nullptr || inputs == nullptr || outputs == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* m = reinterpret_cast<Model*>(model);
    return m->identifyInputsAndOutputs(inputCount, inputs, outputCount, outputs);
}

int ANeuralNetworksModel_relaxComputationFloat32toFloat16(ANeuralNetworksModel* model, bool allow) {
    LOGV(__func__);

    if (model == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* m = reinterpret_cast<Model*>(model);
    return m->relaxComputationFloat32toFloat16(allow);
}

int ANeuralNetworksCompilation_create(ANeuralNetworksModel* model,
                                      ANeuralNetworksCompilation** compilation) {
    LOGV(__func__);
    if (model == nullptr || compilation == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* m = reinterpret_cast<Model*>(model);
    auto* c = new Compilation(m);
    *compilation = reinterpret_cast<ANeuralNetworksCompilation*>(c);
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_createForDevices(ANeuralNetworksModel* model,
                                                const ANeuralNetworksDevice* const* devices,
                                                uint32_t numDevices,
                                                ANeuralNetworksCompilation** compilation) {
    LOGV(__func__);

    if (model == nullptr || devices == nullptr || compilation == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    if (numDevices == 0) {
        LOGE("%s passed an empty device list", __func__);
        return ANEURALNETWORKS_BAD_DATA;
    }

    std::vector<std::shared_ptr<Device>> selectedDevices;
    for (size_t i = 0; i < numDevices; i++) {
        if (devices[i] == nullptr) {
            LOGE("%s passed a nullptr as a device", __func__);
            return ANEURALNETWORKS_UNEXPECTED_NULL;
        }

        for (size_t j = i + 1; j < numDevices; j++) {
            if (devices[i] == devices[j]) {
                LOGE("%s passed duplicate devices", __func__);
                return ANEURALNETWORKS_BAD_DATA;
            }
        }

        for (const auto& device : DeviceManager::get()->getDevices()) {
            if (device.get() == reinterpret_cast<const Device*>(devices[i])) {
                // Found a match.
                selectedDevices.push_back(device);
                break;
            }
        }
    }

    if (selectedDevices.size() != numDevices) {
        LOGE("%s passed an invalid device set", __func__);
    }

    auto* m = reinterpret_cast<Model*>(model);
    auto* c = new Compilation(m, selectedDevices);
    *compilation = reinterpret_cast<ANeuralNetworksCompilation*>(c);
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation* compilation) {
    LOGV(__func__);
    if (compilation == nullptr) {
        LOGV("%s passed a nullptr", __func__);
        return;
    }

    auto* c = reinterpret_cast<Compilation*>(compilation);
    delete c;
}

int ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation* compilation,
                                             int32_t preference) {
    LOGV(__func__);
    if (compilation == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* c = reinterpret_cast<Compilation*>(compilation);
    return c->setPreference(static_cast<PreferenceCode>(preference));
}

int ANeuralNetworksCompilation_setCaching(ANeuralNetworksCompilation* compilation,
                                          const char* cacheDir, const uint8_t* token) {
    LOGV(__func__);
    if (compilation == nullptr || cacheDir == nullptr || token == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* c = reinterpret_cast<Compilation*>(compilation);
    return c->setCaching(cacheDir, token);
}

int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation* compilation) {
    LOGV(__func__);
    if (compilation == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* c = reinterpret_cast<Compilation*>(compilation);
    return c->finish();
}

int ANeuralNetworksCompilation_setPriority(ANeuralNetworksCompilation* compilation, int priority) {
    LOGV(__func__);
    if (compilation == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* c = reinterpret_cast<Compilation*>(compilation);
    return c->setPriority(static_cast<PriorityCode>(priority));
}

int ANeuralNetworksCompilation_setTimeout(ANeuralNetworksCompilation* compilation,
                                          uint64_t duration) {
    LOGV(__func__);
    if (compilation == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    auto* c = reinterpret_cast<Compilation*>(compilation);
    return c->setTimeout(std::chrono::nanoseconds(duration));
}

int ANeuralNetworksExecution_create(ANeuralNetworksCompilation* compilation,
                                    ANeuralNetworksExecution** execution) {
    LOGV(__func__);
    if (compilation == nullptr || execution == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* c = reinterpret_cast<Compilation*>(compilation);
    auto* exec = new Execution(c);
    *execution = reinterpret_cast<ANeuralNetworksExecution*>(exec);
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution) {
    LOGV(__func__);
    if (execution == nullptr) {
        LOGV("%s passed a nullptr", __func__);
        return;
    }

    auto* exec = reinterpret_cast<Execution*>(execution);
    delete exec;
}

int ANeuralNetworksExecution_setReusable(ANeuralNetworksExecution* execution, bool reusable) {
    LOGV(__func__);
    if (execution == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* exec = reinterpret_cast<Execution*>(execution);
    return exec->setReusable(reusable);
}

int ANeuralNetworksExecution_setTimeout(ANeuralNetworksExecution* execution, uint64_t duration) {
    LOGV(__func__);
    if (execution == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* exec = reinterpret_cast<Execution*>(execution);
    return exec->setTimeout(Duration(duration));
}

int ANeuralNetworksExecution_setLoopTimeout(ANeuralNetworksExecution* execution,
                                            uint64_t duration) {
    LOGV(__func__);
    if (execution == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* exec = reinterpret_cast<Execution*>(execution);
    return exec->setLoopTimeout(Duration(duration));
}

int ANeuralNetworksExecution_setMeasureTiming(ANeuralNetworksExecution* execution, bool measure) {
    LOGV(__func__);
    if (execution == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* exec = reinterpret_cast<Execution*>(execution);
    return exec->setMeasureTiming(measure);
}

int ANeuralNetworksExecution_enableInputAndOutputPadding(ANeuralNetworksExecution* execution,
                                                         bool enable) {
    LOGV(__func__);
    if (execution == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution* execution, int32_t index,
                                      const ANeuralNetworksOperandType* type, const void* buffer,
                                      size_t length) {
    LOGV(__func__);
    if (execution == nullptr || (buffer == nullptr && length != 0)) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* exec = reinterpret_cast<Execution*>(execution);
    return exec->setInput(index, type, buffer, length);
}

int ANeuralNetworksExecution_setInputFromMemory(ANeuralNetworksExecution* execution, int32_t index,
                                                const ANeuralNetworksOperandType* type,
                                                const ANeuralNetworksMemory* memory, size_t offset,
                                                size_t length) {
    LOGV(__func__);
    if (execution == nullptr || memory == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* exec = reinterpret_cast<Execution*>(execution);
    const auto* mem = reinterpret_cast<const IMemory*>(memory);
    return exec->setInputFromMemory(index, type, mem, offset, length);
}

int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution* execution, int32_t index,
                                       const ANeuralNetworksOperandType* type, void* buffer,
                                       size_t length) {
    LOGV(__func__);
    if (execution == nullptr || (buffer == nullptr && length != 0)) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* exec = reinterpret_cast<Execution*>(execution);
    return exec->setOutput(index, type, buffer, length);
}

int ANeuralNetworksExecution_setOutputFromMemory(ANeuralNetworksExecution* execution, int32_t index,
                                                 const ANeuralNetworksOperandType* type,
                                                 const ANeuralNetworksMemory* memory, size_t offset,
                                                 size_t length) {
    LOGV(__func__);
    if (execution == nullptr || memory == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* exec = reinterpret_cast<Execution*>(execution);
    const auto* mem = reinterpret_cast<const IMemory*>(memory);
    return exec->setOutputFromMemory(index, type, mem, offset, length);
}

int ANeuralNetworksExecution_compute(ANeuralNetworksExecution* execution) {
    LOGV(__func__);
    if (execution == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* exec = reinterpret_cast<Execution*>(execution);
    return exec->compute();
}

int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution* execution,
                                          ANeuralNetworksEvent** event) {
    LOGV(__func__);
    if (event == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    if (execution == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        *event = nullptr;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* exec = reinterpret_cast<Execution*>(execution);
    auto* e = exec->createSyncEvent();
    *event = reinterpret_cast<ANeuralNetworksEvent*>(e);

    return exec->startCompute();
}

int ANeuralNetworksExecution_getDuration(const ANeuralNetworksExecution* execution,
                                         int32_t durationCode, uint64_t* duration) {
    LOGV(__func__);
    if (execution == nullptr || duration == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    const auto* exec = reinterpret_cast<const Execution*>(execution);
    return exec->getDuration(static_cast<DurationCode>(durationCode), duration);
}

int ANeuralNetworksExecution_getOutputOperandRank(ANeuralNetworksExecution* execution,
                                                  int32_t index, uint32_t* rank) {
    LOGV(__func__);
    if (rank == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    const auto* exec = reinterpret_cast<const Execution*>(execution);
    return exec->getOutputOperandRank(index, rank);
}

int ANeuralNetworksExecution_getOutputOperandDimensions(ANeuralNetworksExecution* execution,
                                                        int32_t index, uint32_t* dimensions) {
    LOGV(__func__);
    if (dimensions == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    const auto* exec = reinterpret_cast<const Execution*>(execution);
    return exec->getOutputOperandDimensions(index, dimensions);
}

int ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event) {
    LOGV(__func__);

    if (event == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* e = reinterpret_cast<IEvent*>(event);
    return e->wait();
}

void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event) {
    LOGV(__func__);
    if (event == nullptr) {
        LOGD("%s passed a nullptr", __func__);
        return;
    }

    auto* e = reinterpret_cast<IEvent*>(event);
    e->wait();
    delete e;
}

uint64_t ANeuralNetworks_getDefaultLoopTimeout() {
    LOGV(__func__);

    constexpr auto kDefaultLoopTimeoutDuration = std::chrono::seconds{2};
    constexpr uint64_t kDefaultLoopTimeoutNs =
            std::chrono::duration_cast<std::chrono::nanoseconds>(kDefaultLoopTimeoutDuration)
                    .count();

    return kDefaultLoopTimeoutNs;
}

uint64_t ANeuralNetworks_getMaximumLoopTimeout() {
    LOGV(__func__);

    constexpr auto kMaximumLoopTimeoutDuration = std::chrono::seconds{15};
    constexpr uint64_t kMaximumLoopTimeoutNs =
            std::chrono::duration_cast<std::chrono::nanoseconds>(kMaximumLoopTimeoutDuration)
                    .count();

    return kMaximumLoopTimeoutNs;
}

int ANeuralNetworksDevice_getExtensionSupport(const ANeuralNetworksDevice* device,
                                              const char* extensionName,
                                              bool* isExtensionSupported) {
    LOGV(__func__);

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_getExtensionOperandType(ANeuralNetworksModel* model,
                                                 const char* extensionName,
                                                 uint16_t operandCodeWithinExtension,
                                                 int32_t* type) {
    LOGV(__func__);

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_getExtensionOperationType(ANeuralNetworksModel* model,
                                                   const char* extensionName,
                                                   uint16_t operationCodeWithinExtension,
                                                   ANeuralNetworksOperationType* type) {
    LOGV(__func__);

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_setOperandExtensionData(ANeuralNetworksModel* model, int32_t index,
                                                 const void* data, size_t length) {
    LOGV(__func__);

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksEvent_createFromSyncFenceFd(int sync_fence_fd, ANeuralNetworksEvent** event) {
    LOGV(__func__);

    if (event == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    if (sync_fence_fd <= 0) {
        LOGE("%s passed an invalid sync fence fd", __func__);
        *event = nullptr;
        return ANEURALNETWORKS_BAD_DATA;
    }

    auto* e = new SyncFenceEvent(sync_fence_fd);
    *event = reinterpret_cast<ANeuralNetworksEvent*>(e);

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksEvent_getSyncFenceFd(const ANeuralNetworksEvent* event, int* sync_fence_fd) {
    LOGV(__func__);

    if (sync_fence_fd == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    if (event == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        *sync_fence_fd = -1;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    const auto* e = reinterpret_cast<const IEvent*>(event);
    // The client owns the dupped fd, and is responsible for closing it.
    int fd = e->getSyncFenceFd(true);
    if (fd <= 0) {
        LOGE("%s unable to get valid sync fence fd", __func__);
        *sync_fence_fd = -1;
        return ANEURALNETWORKS_BAD_DATA;
    }

    *sync_fence_fd = fd;

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_startComputeWithDependencies(
        ANeuralNetworksExecution* execution, const ANeuralNetworksEvent* const* dependencies,
        uint32_t num_dependencies, uint64_t duration, ANeuralNetworksEvent** event) {
    LOGV(__func__);

    if (event == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    if (execution == nullptr || (num_dependencies != 0 && dependencies == nullptr)) {
        LOGE("%s passed a nullptr", __func__);
        *event = nullptr;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    auto* exec = reinterpret_cast<Execution*>(execution);

    if (duration != 0) {
        const auto* compilation = exec->getCompilation();
        if (compilation->getDevices().size() != 1) {
            LOGE("%s if the duration is non-zero, the "
                 "ANeuralNetworksExecution must have been created from an "
                 "ANeuralNetworksCompilation which in turn was created from "
                 "ANeuralNetworksCompilation_createForDevices with numDevices = 1",
                 __func__);
            return ANEURALNETWORKS_BAD_DATA;
        }
    }

    for (size_t i = 0; i < num_dependencies; i++) {
        if (dependencies[i] == nullptr) {
            LOGE("%s passed a nullptr", __func__);
            *event = nullptr;
            return ANEURALNETWORKS_UNEXPECTED_NULL;
        }

        const auto* e = reinterpret_cast<const IEvent*>(dependencies[i]);
        int waitStatus = e->wait();
        if (waitStatus != ANEURALNETWORKS_NO_ERROR) {
            *event = nullptr;
            return waitStatus;
        }
    }

    // The SL don't support creating sync fence.
    auto* e = new SyncFenceEvent(-1);
    *event = reinterpret_cast<ANeuralNetworksEvent*>(e);
    return ANeuralNetworksExecution_compute(execution);

    // return ANeuralNetworksExecution_startCompute(execution, event);
}

int64_t ANeuralNetworks_getRuntimeFeatureLevel() {
    LOGV(__func__);

    return ANEURALNETWORKS_FEATURE_LEVEL_7;
}

int ANeuralNetworksCompilation_getPreferredMemoryAlignmentForInput(
        const ANeuralNetworksCompilation* compilation, uint32_t index, uint32_t* alignment) {
    LOGV(__func__);

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_getPreferredMemoryPaddingForInput(
        const ANeuralNetworksCompilation* compilation, uint32_t index, uint32_t* padding) {
    LOGV(__func__);

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput(
        const ANeuralNetworksCompilation* compilation, uint32_t index, uint32_t* alignment) {
    LOGV(__func__);

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput(
        const ANeuralNetworksCompilation* compilation, uint32_t index, uint32_t* padding) {
    LOGV(__func__);

    return ANEURALNETWORKS_NO_ERROR;
}

int SL_ANeuralNetworksCompilation_setCachingFromFds(ANeuralNetworksCompilation* compilation,
                                                    const int* modelCacheFds,
                                                    const uint32_t numModelCacheFiles,
                                                    const int* /*dataCacheFds*/,
                                                    const uint32_t /*numDataCacheFiles*/,
                                                    const uint8_t* token) {
    LOGV(__func__);
    if (compilation == nullptr ||
        (numModelCacheFiles != 0 && (modelCacheFds == nullptr || token == nullptr))) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    if (Compilation::kNumModelCacheFiles == 0) {
        LOGW("%s model cache is not enabled", __func__);
        return ANEURALNETWORKS_NO_ERROR;
    }

    auto* c = reinterpret_cast<Compilation*>(compilation);
    return c->setCaching(modelCacheFds[0], token);
}

int SL_ANeuralNetworksDevice_getNumberOfCacheFilesNeeded(const ANeuralNetworksDevice* device,
                                                         uint32_t* numModelCacheFiles,
                                                         uint32_t* numDataCacheFiles) {
    LOGV(__func__);
    if (numDataCacheFiles == nullptr || numDataCacheFiles == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    *numModelCacheFiles = Compilation::kNumModelCacheFiles;
    *numDataCacheFiles = Compilation::kNumDataCacheFiles;

    return ANEURALNETWORKS_NO_ERROR;
}

int SL_ANeuralNetworksDevice_getPerformanceInfo(
        const ANeuralNetworksDevice* device, int32_t performanceInfoKind,
        SL_ANeuralNetworksPerformanceInfo* performanceInfo) {
    LOGV(__func__);

    if (device == nullptr || performanceInfo == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    const auto* dev = reinterpret_cast<const Device*>(device);
    auto perfInfo = dev->queryPerformanceInfo(performanceInfoKind);

    performanceInfo->execTime = perfInfo.execTimeRatio;
    performanceInfo->powerUsage = perfInfo.powerUsageRatio;
    return ANEURALNETWORKS_NO_ERROR;
}

int SL_ANeuralNetworksDevice_forEachOperandPerformanceInfo(
        const ANeuralNetworksDevice* device, void* context,
        void (*callback)(SL_ANeuralNetworksOperandPerformanceInfo, void*)) {
    LOGV(__func__);

    if (device == nullptr || context == nullptr || callback == nullptr) {
        LOGE("%s passed a nullptr", __func__);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    const auto* dev = reinterpret_cast<const Device*>(device);
    for (auto operandType : Device::kSupportedOperandTypes) {
        auto perfInfo = dev->queryOperandPerformanceInfo(operandType);
        auto operandPerformanceInfo = SL_ANeuralNetworksOperandPerformanceInfo{
                .operandType = static_cast<int32_t>(operandType),
                .performanceInfo = {
                        .execTime = perfInfo.execTimeRatio,
                        .powerUsage = perfInfo.powerUsageRatio,
                }};
        callback(operandPerformanceInfo, context);
    }

    return ANEURALNETWORKS_NO_ERROR;
}

int SL_ANeuralNetworksDevice_getVendorExtensionCount(const ANeuralNetworksDevice* device,
                                                     uint32_t* vendorExtensionCount) {
    LOGV(__func__);
    if (device == nullptr || vendorExtensionCount == nullptr) {
        LOGE("SL_ANeuralNetworksDevice_getVendorExtensionCount passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    *vendorExtensionCount = 0;
    return ANEURALNETWORKS_NO_ERROR;
}

int SL_ANeuralNetworksDevice_getVendorExtensionName(const ANeuralNetworksDevice* device,
                                                    uint32_t vendorExtensionIndex,
                                                    const char** extensionName) {
    LOGV(__func__);

    return ANEURALNETWORKS_NO_ERROR;
}

int SL_ANeuralNetworksDevice_forEachVendorExtensionOperandTypeInformation(
        const ANeuralNetworksDevice* device, uint32_t vendorExtensionIndex, void* context,
        void (*callback)(SL_ANeuralNetworksExtensionOperandTypeInformation, void*)) {
    LOGV(__func__);

    return ANEURALNETWORKS_NO_ERROR;
}

#define NNCL_FUNC(symbol) .symbol = symbol

NnApiSLDriverImplFL7 slDriverImpl{
        .base{.implFeatureLevel = ANEURALNETWORKS_FEATURE_LEVEL_7},
        NNCL_FUNC(ANeuralNetworksBurst_create),
        NNCL_FUNC(ANeuralNetworksBurst_free),
        NNCL_FUNC(ANeuralNetworksCompilation_createForDevices),
        NNCL_FUNC(ANeuralNetworksCompilation_finish),
        NNCL_FUNC(ANeuralNetworksCompilation_free),
        NNCL_FUNC(ANeuralNetworksCompilation_getPreferredMemoryAlignmentForInput),
        NNCL_FUNC(ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput),
        NNCL_FUNC(ANeuralNetworksCompilation_getPreferredMemoryPaddingForInput),
        NNCL_FUNC(ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput),
        NNCL_FUNC(ANeuralNetworksCompilation_setCaching),
        NNCL_FUNC(ANeuralNetworksCompilation_setPreference),
        NNCL_FUNC(ANeuralNetworksCompilation_setPriority),
        NNCL_FUNC(ANeuralNetworksCompilation_setTimeout),
        NNCL_FUNC(ANeuralNetworksDevice_getExtensionSupport),
        NNCL_FUNC(ANeuralNetworksDevice_getFeatureLevel),
        NNCL_FUNC(ANeuralNetworksDevice_getName),
        NNCL_FUNC(ANeuralNetworksDevice_getType),
        NNCL_FUNC(ANeuralNetworksDevice_getVersion),
        NNCL_FUNC(ANeuralNetworksDevice_wait),
        NNCL_FUNC(ANeuralNetworksEvent_createFromSyncFenceFd),
        NNCL_FUNC(ANeuralNetworksEvent_free),
        NNCL_FUNC(ANeuralNetworksEvent_getSyncFenceFd),
        NNCL_FUNC(ANeuralNetworksEvent_wait),
        NNCL_FUNC(ANeuralNetworksExecution_burstCompute),
        NNCL_FUNC(ANeuralNetworksExecution_compute),
        NNCL_FUNC(ANeuralNetworksExecution_create),
        NNCL_FUNC(ANeuralNetworksExecution_enableInputAndOutputPadding),
        NNCL_FUNC(ANeuralNetworksExecution_free),
        NNCL_FUNC(ANeuralNetworksExecution_getDuration),
        NNCL_FUNC(ANeuralNetworksExecution_getOutputOperandDimensions),
        NNCL_FUNC(ANeuralNetworksExecution_getOutputOperandRank),
        NNCL_FUNC(ANeuralNetworksExecution_setInput),
        NNCL_FUNC(ANeuralNetworksExecution_setInputFromMemory),
        NNCL_FUNC(ANeuralNetworksExecution_setLoopTimeout),
        NNCL_FUNC(ANeuralNetworksExecution_setMeasureTiming),
        NNCL_FUNC(ANeuralNetworksExecution_setOutput),
        NNCL_FUNC(ANeuralNetworksExecution_setOutputFromMemory),
        NNCL_FUNC(ANeuralNetworksExecution_setReusable),
        NNCL_FUNC(ANeuralNetworksExecution_setTimeout),
        NNCL_FUNC(ANeuralNetworksExecution_startComputeWithDependencies),
        NNCL_FUNC(ANeuralNetworksMemoryDesc_addInputRole),
        NNCL_FUNC(ANeuralNetworksMemoryDesc_addOutputRole),
        NNCL_FUNC(ANeuralNetworksMemoryDesc_create),
        NNCL_FUNC(ANeuralNetworksMemoryDesc_finish),
        NNCL_FUNC(ANeuralNetworksMemoryDesc_free),
        NNCL_FUNC(ANeuralNetworksMemoryDesc_setDimensions),
        NNCL_FUNC(ANeuralNetworksMemory_copy),
        NNCL_FUNC(ANeuralNetworksMemory_createFromAHardwareBuffer),
        NNCL_FUNC(ANeuralNetworksMemory_createFromDesc),
        NNCL_FUNC(ANeuralNetworksMemory_createFromFd),
        NNCL_FUNC(ANeuralNetworksMemory_free),
        NNCL_FUNC(ANeuralNetworksModel_addOperand),
        NNCL_FUNC(ANeuralNetworksModel_addOperation),
        NNCL_FUNC(ANeuralNetworksModel_create),
        NNCL_FUNC(ANeuralNetworksModel_finish),
        NNCL_FUNC(ANeuralNetworksModel_free),
        NNCL_FUNC(ANeuralNetworksModel_getExtensionOperandType),
        NNCL_FUNC(ANeuralNetworksModel_getExtensionOperationType),
        NNCL_FUNC(ANeuralNetworksModel_getSupportedOperationsForDevices),
        NNCL_FUNC(ANeuralNetworksModel_identifyInputsAndOutputs),
        NNCL_FUNC(ANeuralNetworksModel_relaxComputationFloat32toFloat16),
        NNCL_FUNC(ANeuralNetworksModel_setOperandExtensionData),
        NNCL_FUNC(ANeuralNetworksModel_setOperandSymmPerChannelQuantParams),
        NNCL_FUNC(ANeuralNetworksModel_setOperandValue),
        NNCL_FUNC(ANeuralNetworksModel_setOperandValueFromMemory),
        NNCL_FUNC(ANeuralNetworksModel_setOperandValueFromModel),
        NNCL_FUNC(ANeuralNetworks_getDefaultLoopTimeout),
        NNCL_FUNC(ANeuralNetworks_getDevice),
        NNCL_FUNC(ANeuralNetworks_getDeviceCount),
        NNCL_FUNC(ANeuralNetworks_getMaximumLoopTimeout),
        NNCL_FUNC(ANeuralNetworks_getRuntimeFeatureLevel),
        NNCL_FUNC(SL_ANeuralNetworksCompilation_setCachingFromFds),
        NNCL_FUNC(SL_ANeuralNetworksDevice_getNumberOfCacheFilesNeeded),
        NNCL_FUNC(SL_ANeuralNetworksDevice_getPerformanceInfo),
        NNCL_FUNC(SL_ANeuralNetworksDevice_forEachOperandPerformanceInfo),
        NNCL_FUNC(SL_ANeuralNetworksDevice_getVendorExtensionCount),
        NNCL_FUNC(SL_ANeuralNetworksDevice_getVendorExtensionName),
        NNCL_FUNC(SL_ANeuralNetworksDevice_forEachVendorExtensionOperandTypeInformation),
        .SL_ANeuralNetworksDiagnosticCompilationInfo_getSessionId = nullptr,
        .SL_ANeuralNetworksDiagnosticCompilationInfo_getNnApiVersion = nullptr,
        .SL_ANeuralNetworksDiagnosticCompilationInfo_getModelArchHash = nullptr,
        .SL_ANeuralNetworksDiagnosticCompilationInfo_getDeviceIds = nullptr,
        .SL_ANeuralNetworksDiagnosticCompilationInfo_getErrorCode = nullptr,
        .SL_ANeuralNetworksDiagnosticCompilationInfo_getInputDataClass = nullptr,
        .SL_ANeuralNetworksDiagnosticCompilationInfo_getOutputDataClass = nullptr,
        .SL_ANeuralNetworksDiagnosticCompilationInfo_getCompilationTimeNanos = nullptr,
        .SL_ANeuralNetworksDiagnosticCompilationInfo_isCachingEnabled = nullptr,
        .SL_ANeuralNetworksDiagnosticCompilationInfo_isControlFlowUsed = nullptr,
        .SL_ANeuralNetworksDiagnosticCompilationInfo_areDynamicTensorsUsed = nullptr,
        .SL_ANeuralNetworksDiagnosticExecutionInfo_getSessionId = nullptr,
        .SL_ANeuralNetworksDiagnosticExecutionInfo_getNnApiVersion = nullptr,
        .SL_ANeuralNetworksDiagnosticExecutionInfo_getModelArchHash = nullptr,
        .SL_ANeuralNetworksDiagnosticExecutionInfo_getDeviceIds = nullptr,
        .SL_ANeuralNetworksDiagnosticExecutionInfo_getExecutionMode = nullptr,
        .SL_ANeuralNetworksDiagnosticExecutionInfo_getInputDataClass = nullptr,
        .SL_ANeuralNetworksDiagnosticExecutionInfo_getOutputDataClass = nullptr,
        .SL_ANeuralNetworksDiagnosticExecutionInfo_getErrorCode = nullptr,
        .SL_ANeuralNetworksDiagnosticExecutionInfo_getRuntimeExecutionTimeNanos = nullptr,
        .SL_ANeuralNetworksDiagnosticExecutionInfo_getHardwareExecutionTimeNanos = nullptr,
        .SL_ANeuralNetworksDiagnosticExecutionInfo_isCachingEnabled = nullptr,
        .SL_ANeuralNetworksDiagnosticExecutionInfo_isControlFlowUsed = nullptr,
        .SL_ANeuralNetworksDiagnosticExecutionInfo_areDynamicTensorsUsed = nullptr,
        .SL_ANeuralNetworksDiagnostic_registerCallbacks = nullptr};

#undef NNCL_FUNC

extern "C" {
NnApiSLDriverImpl* ANeuralNetworks_getSLDriverImpl() {
    LOGV(__func__);
    return reinterpret_cast<NnApiSLDriverImpl*>(&slDriverImpl);
}
}