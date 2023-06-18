/****************************************************************************
 *
 *    Copyright (c) 2023 Vivante Corporation
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
#include <android/log.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "Compilation.h"
#include "DeviceManager.h"
#include "Execution.h"
#include "Memory.h"
#include "Model.h"
#include "NeuralNetworksSupportLibrary.h"
#include "VsiDevice.h"

// using namespace android::nn;
using namespace vsi::android::sl;
#define TAG_NAME "NNAPI-SL"

int ANeuralNetworks_getDeviceCount(uint32_t* numDevices) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworks_getDeviceCount is called ");

    *numDevices = DeviceManager::Instance()->GetDevices().size();
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworks_getDevice(uint32_t devIndex, ANeuralNetworksDevice** device) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME, "=====ANeuralNetworks_getDevice is called ");

    if (device == nullptr) {
        __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                            "ANeuralNetworks_getDevice passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    auto devices = DeviceManager::Instance()->GetDevices();
    *device = reinterpret_cast<ANeuralNetworksDevice*>(devices.at(devIndex).get());

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksDevice_getName(const ANeuralNetworksDevice* device, const char** name) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksDevice_getName is called ");

    if (device == nullptr || name == nullptr) {
        __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                            "ANeuralNetworksDevice_getName passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    const VsiDevice* d = reinterpret_cast<const VsiDevice*>(device);

    *name = d->GetName().c_str();
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksDevice_getVersion(const ANeuralNetworksDevice* device, const char** version) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksDevice_getVersion is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksDevice_getType(const ANeuralNetworksDevice* device, int32_t* type) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksDevice_getType is called ");

    *type = 4;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksDevice_getFeatureLevel(const ANeuralNetworksDevice* device,
                                          int64_t* featureLevel) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksDevice_getFeatureLevel is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksDevice_wait(const ANeuralNetworksDevice* device) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksDevice_wait is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_getSupportedOperationsForDevices(
        const ANeuralNetworksModel* model, const ANeuralNetworksDevice* const* devices,
        uint32_t numDevices, bool* supportedOps) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksModel_getSupportedOperationsForDevices is "
                        "called ");

    if (!model || !devices || !supportedOps) {
        __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                            "ANeuralNetworksModel_getSupportedOperationsForDevices get nullptr");

        return ANEURALNETWORKS_BAD_DATA;
    }
    const Model* m = reinterpret_cast<const Model*>(model);
    return m->GetSupportedOperations(supportedOps);
}

int ANeuralNetworksCompilation_createForDevices(ANeuralNetworksModel* model,
                                                const ANeuralNetworksDevice* const* devices,
                                                uint32_t numDevices,
                                                ANeuralNetworksCompilation** compilation) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksCompilation_createForDevices is called ");

    Model* m = reinterpret_cast<Model*>(model);
    // pointer "devices" may be released after this call, so we must copy its
    // content here.
    const VsiDevice* const* vsiDevices = reinterpret_cast<const VsiDevice* const*>(devices);
    std::vector<const VsiDevice*> vsiDeviceVec;
    for (uint32_t i = 0; i < numDevices; ++i) vsiDeviceVec.push_back(vsiDevices[i]);

    Compilation* c = new Compilation(m, vsiDeviceVec);
    *compilation = reinterpret_cast<ANeuralNetworksCompilation*>(c);
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_compute(ANeuralNetworksExecution* execution) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_compute is called ");

    Execution* e = reinterpret_cast<Execution*>(execution);
    return e->Compute();
}

int ANeuralNetworksExecution_setMeasureTiming(ANeuralNetworksExecution* execution, bool measure) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_setMeasureTiming is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_getDuration(const ANeuralNetworksExecution* execution,
                                         int32_t durationCode, uint64_t* duration) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_getDuration is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksBurst_create(ANeuralNetworksCompilation* compilation,
                                ANeuralNetworksBurst** burst) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksBurst_create is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksBurst_free(ANeuralNetworksBurst* burst) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME, "=====ANeuralNetworksBurst_free is called ");
}

int ANeuralNetworksExecution_burstCompute(ANeuralNetworksExecution* execution,
                                          ANeuralNetworksBurst* burst) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_burstCompute is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksMemoryDesc_create(ANeuralNetworksMemoryDesc** desc) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksMemoryDesc_create is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksMemoryDesc_free(ANeuralNetworksMemoryDesc* desc) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksMemoryDesc_free is called ");
}

int ANeuralNetworksMemoryDesc_addInputRole(ANeuralNetworksMemoryDesc* desc,
                                           const ANeuralNetworksCompilation* compilation,
                                           uint32_t index, float frequency) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksMemoryDesc_addInputRole is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksMemoryDesc_addOutputRole(ANeuralNetworksMemoryDesc* desc,
                                            const ANeuralNetworksCompilation* compilation,
                                            uint32_t index, float frequency) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksMemoryDesc_addOutputRole is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksMemoryDesc_setDimensions(ANeuralNetworksMemoryDesc* desc, uint32_t rank,
                                            const uint32_t* dimensions) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksMemoryDesc_setDimensions is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksMemoryDesc_finish(ANeuralNetworksMemoryDesc* desc) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksMemoryDesc_finish is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksMemory_createFromDesc(const ANeuralNetworksMemoryDesc* desc,
                                         ANeuralNetworksMemory** memory) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksMemory_createFromDesc is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksMemory_copy(const ANeuralNetworksMemory* src, const ANeuralNetworksMemory* dst) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksMemory_copy is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksMemory_createFromFd(size_t size, int prot, int fd, size_t offset,
                                       ANeuralNetworksMemory** memory) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksMemory_createFromFd is called ");

    Memory* mem = new Memory();
    mem->CreateFromFd(size, prot, fd, offset);
    *memory = reinterpret_cast<ANeuralNetworksMemory*>(mem);
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksMemory_createFromAHardwareBuffer(const AHardwareBuffer* ahwb,
                                                    ANeuralNetworksMemory** memory) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksMemory_createFromAHardwareBuffer is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksMemory_free is called ");

    if (memory == nullptr) return;
    Memory* mem = reinterpret_cast<Memory*>(memory);
    delete mem;
    mem = nullptr;
}

int ANeuralNetworksModel_create(ANeuralNetworksModel** model) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksModel_create is called ");

    if (!model) {
        __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                            "ANeuralNetworksModel_create passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    Model* m = new Model();
    *model = reinterpret_cast<ANeuralNetworksModel*>(m);
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksModel_free(ANeuralNetworksModel* model) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME, "=====ANeuralNetworksModel_free is called ");

    if (model == nullptr) return;
    Model* m = reinterpret_cast<Model*>(model);
    delete m;
    m = nullptr;
}

int ANeuralNetworksModel_finish(ANeuralNetworksModel* model) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksModel_finish is called ");

    Model* m = reinterpret_cast<Model*>(model);
    return m->Finish();
}

int ANeuralNetworksModel_addOperand(ANeuralNetworksModel* model,
                                    const ANeuralNetworksOperandType* type) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksModel_addOperand is called ");

    Model* m = reinterpret_cast<Model*>(model);
    return m->AddOperand(*type);
}

int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model, int32_t index,
                                         const void* buffer, size_t length) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksModel_setOperandValue is called ");

    if (!model || (!buffer && length != 0)) {
        __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                            "ANeuralNetworksModel_setOperandValue passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    Model* m = reinterpret_cast<Model*>(model);
    return m->SetOperandValue(index, buffer, length);
}

int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel* model, int32_t index,
                                                   const ANeuralNetworksMemory* memory,
                                                   size_t offset, size_t length) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksModel_setOperandValueFromMemory is called ");

    Model* m = reinterpret_cast<Model*>(model);
    const Memory* mem = reinterpret_cast<const Memory*>(memory);
    return m->SetOperandValueFromMemory(index, mem, offset, length);
}

int ANeuralNetworksModel_setOperandValueFromModel(ANeuralNetworksModel* model, int32_t index,
                                                  const ANeuralNetworksModel* value) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksModel_setOperandValueFromModel is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                      ANeuralNetworksOperationType type, uint32_t inputCount,
                                      const uint32_t* inputs, uint32_t outputCount,
                                      const uint32_t* outputs) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksModel_addOperation is called ");

    if (!model || !inputs || !outputs) {
        __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                            "ANeuralNetworksModel_addOperation passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    Model* m = reinterpret_cast<Model*>(model);
    return m->AddOperation(type, inputCount, inputs, outputCount, outputs);
}

int ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(
        ANeuralNetworksModel* model, int32_t index,
        const ANeuralNetworksSymmPerChannelQuantParams* channelQuant) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksModel_setOperandSymmPerChannelQuantParams "
                        "is called ");

    Model* m = reinterpret_cast<Model*>(model);
    return m->SetOperandSymmPerChannelQuantParams(index, *channelQuant);
}

int ANeuralNetworksModel_identifyInputsAndOutputs(ANeuralNetworksModel* model, uint32_t inputCount,
                                                  const uint32_t* inputs, uint32_t outputCount,
                                                  const uint32_t* outputs) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksModel_identifyInputsAndOutputs is called ");

    if (!model || !inputs || !outputs) {
        __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                            "ANeuralNetworksModel_identifyInputsAndOutputs passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    Model* m = reinterpret_cast<Model*>(model);
    return m->IdentifyInputsAndOutputs(inputCount, inputs, outputCount, outputs);
}

int ANeuralNetworksModel_relaxComputationFloat32toFloat16(ANeuralNetworksModel* model, bool allow) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksModel_relaxComputationFloat32toFloat16 is "
                        "called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_create(ANeuralNetworksModel* model,
                                      ANeuralNetworksCompilation** compilation) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksCompilation_create is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation* compilation) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksCompilation_free is called ");

    if (compilation == nullptr) {
        return;
    }
    Compilation* c = reinterpret_cast<Compilation*>(compilation);
    delete c;
    c = nullptr;
}

int ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation* compilation,
                                             int32_t preference) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksCompilation_setPreference is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_setCaching(ANeuralNetworksCompilation* compilation,
                                          const char* cacheDir, const uint8_t* token) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksCompilation_setCaching is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation* compilation) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksCompilation_finish is called ");

    Compilation* c = reinterpret_cast<Compilation*>(compilation);
    return c->Finish();
}

int ANeuralNetworksCompilation_setPriority(ANeuralNetworksCompilation* compilation, int priority) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksCompilation_setPrioritytworks");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_setTimeout(ANeuralNetworksCompilation* compilation,
                                          uint64_t duration) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksCompilation_setTimeout is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_create(ANeuralNetworksCompilation* compilation,
                                    ANeuralNetworksExecution** execution) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_create is called ");

    Compilation* c = reinterpret_cast<Compilation*>(compilation);
    Execution* e = new Execution(c);
    *execution = reinterpret_cast<ANeuralNetworksExecution*>(e);
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_free is called ");

    if (execution == nullptr) return;
    Execution* e = reinterpret_cast<Execution*>(execution);
    delete e;
    e = nullptr;
}

int ANeuralNetworksExecution_getOutputOperandRank(ANeuralNetworksExecution* execution,
                                                  int32_t index, uint32_t* rank) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_getOutputOperandRank is called ");

    if (rank == nullptr) {
        __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                            "ANeuralNetworksExecution_getOutputOperandRank get a nullptr");

        return ANEURALNETWORKS_BAD_DATA;
    }
    Execution* e = reinterpret_cast<Execution*>(execution);
    return e->GetOutputOperandRank(index, rank);
}

int ANeuralNetworksExecution_getOutputOperandDimensions(ANeuralNetworksExecution* execution,
                                                        int32_t index, uint32_t* dimensions) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_getOutputOperandDimensions is called ");

    if (dimensions == nullptr) {
        __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                            "ANeuralNetworksExecution_getOutputOperandDimensions get a nullptr");

        return ANEURALNETWORKS_BAD_DATA;
    }
    Execution* e = reinterpret_cast<Execution*>(execution);
    return e->GetOutputOperandDimensions(index, dimensions);
}

int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution* execution, int32_t index,
                                      const ANeuralNetworksOperandType* type, const void* buffer,
                                      size_t length) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_setInput is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setInputFromMemory(ANeuralNetworksExecution* execution, int32_t index,
                                                const ANeuralNetworksOperandType* type,
                                                const ANeuralNetworksMemory* memory, size_t offset,
                                                size_t length) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_setInputFromMemory is called ");

    Execution* e = reinterpret_cast<Execution*>(execution);
    const Memory* mem = reinterpret_cast<const Memory*>(memory);
    return e->SetInputFromMemory(index, type, mem, offset, length);
}

int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution* execution, int32_t index,
                                       const ANeuralNetworksOperandType* type, void* buffer,
                                       size_t length) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_setOutput is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setOutputFromMemory(ANeuralNetworksExecution* execution, int32_t index,
                                                 const ANeuralNetworksOperandType* type,
                                                 const ANeuralNetworksMemory* memory, size_t offset,
                                                 size_t length) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_setOutputFromMemory is called ");

    Execution* e = reinterpret_cast<Execution*>(execution);
    const Memory* mem = reinterpret_cast<const Memory*>(memory);
    return e->SetOutputFromMemory(index, type, mem, offset, length);
}

int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution* execution,
                                          ANeuralNetworksEvent** event) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_startCompute is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setTimeout(ANeuralNetworksExecution* execution, uint64_t duration) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_setTimeout is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME, "=====ANeuralNetworksEvent_wait is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME, "=====ANeuralNetworksEvent_free is called ");
}

int ANeuralNetworksExecution_setLoopTimeout(ANeuralNetworksExecution* execution,
                                            uint64_t duration) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_setLoopTimeout is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

uint64_t ANeuralNetworks_getDefaultLoopTimeout() {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworks_getDefaultLoopTimeout is called ");

    return operation_while::kTimeoutNsDefault;
}

uint64_t ANeuralNetworks_getMaximumLoopTimeout() {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworks_getMaximumLoopTimeout is called ");

    return operation_while::kTimeoutNsMaximum;
}

int ANeuralNetworksDevice_getExtensionSupport(const ANeuralNetworksDevice* device,
                                              const char* extensionName,
                                              bool* isExtensionSupported) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksDevice_getExtensionSupport is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_getExtensionOperandType(ANeuralNetworksModel* model,
                                                 const char* extensionName,
                                                 uint16_t operandCodeWithinExtension,
                                                 int32_t* type) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksModel_getExtensionOperandType is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_getExtensionOperationType(ANeuralNetworksModel* model,
                                                   const char* extensionName,
                                                   uint16_t operationCodeWithinExtension,
                                                   ANeuralNetworksOperationType* type) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksModel_getExtensionOperationType is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_setOperandExtensionData(ANeuralNetworksModel* model, int32_t index,
                                                 const void* data, size_t length) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksModel_setOperandExtensionData is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksEvent_createFromSyncFenceFd(int syncFenceFd, ANeuralNetworksEvent** event) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksEvent_createFromSyncFenceFd is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksEvent_getSyncFenceFd(const ANeuralNetworksEvent* event, int* syncFenceFd) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksEvent_getSyncFenceFd is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_startComputeWithDependencies(
        ANeuralNetworksExecution* execution, const ANeuralNetworksEvent* const* dependencies,
        uint32_t numOfDependencies, uint64_t duration, ANeuralNetworksEvent** event) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_startComputeWithDependencies is "
                        "called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int64_t ANeuralNetworks_getRuntimeFeatureLevel() {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworks_getRuntimeFeatureLevel is called ");

    return ANEURALNETWORKS_FEATURE_LEVEL_5;
}

int ANeuralNetworksExecution_enableInputAndOutputPadding(ANeuralNetworksExecution* execution,
                                                         bool enable) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_enableInputAndOutputPadding is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_getPreferredMemoryAlignmentForInput(
        const ANeuralNetworksCompilation* compilation, uint32_t index, uint32_t* alignment) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksCompilation_"
                        "getPreferredMemoryAlignmentForInput is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_getPreferredMemoryPaddingForInput(
        const ANeuralNetworksCompilation* compilation, uint32_t index, uint32_t* padding) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksCompilation_"
                        "getPreferredMemoryPaddingForInput is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput(
        const ANeuralNetworksCompilation* compilation, uint32_t index, uint32_t* alignment) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksCompilation_"
                        "getPreferredMemoryAlignmentForOutput is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput(
        const ANeuralNetworksCompilation* compilation, uint32_t index, uint32_t* padding) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksCompilation_"
                        "getPreferredMemoryPaddingForOutput is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setReusable(ANeuralNetworksExecution* execution, bool reusable) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====ANeuralNetworksExecution_setReusable is called ");

    Execution* e = reinterpret_cast<Execution*>(execution);
    return e->SetReusable(reusable);
}

int SL_ANeuralNetworksCompilation_setCachingFromFds(ANeuralNetworksCompilation* compilation,
                                                    const int* modelCacheFds,
                                                    const uint32_t numModelCacheFiles,
                                                    const int* dataCacheFds,
                                                    const uint32_t numDataCacheFiles,
                                                    const uint8_t* token) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====SL_ANeuralNetworksCompilation_setCachingFromFds is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int SL_ANeuralNetworksDevice_getNumberOfCacheFilesNeeded(const ANeuralNetworksDevice* device,
                                                         uint32_t* numModelCacheFiles,
                                                         uint32_t* numDataCacheFiles) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====SL_ANeuralNetworksDevice_getNumberOfCacheFilesNeeded is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int SL_ANeuralNetworksDevice_getPerformanceInfo(
        const ANeuralNetworksDevice* device, int32_t performanceInfoKind,
        SL_ANeuralNetworksPerformanceInfo* performanceInfo) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====SL_ANeuralNetworksDevice_getPerformanceInfo is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int SL_ANeuralNetworksDevice_forEachOperandPerformanceInfo(
        const ANeuralNetworksDevice* device, void* context,
        void (*callback)(SL_ANeuralNetworksOperandPerformanceInfo, void*)) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====SL_ANeuralNetworksDevice_forEachOperandPerformanceInfo "
                        "is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int SL_ANeuralNetworksDevice_getVendorExtensionCount(const ANeuralNetworksDevice* device,
                                                     uint32_t* vendorExtensionCount) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====SL_ANeuralNetworksDevice_getVendorExtensionCount is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int SL_ANeuralNetworksDevice_getVendorExtensionName(const ANeuralNetworksDevice* device,
                                                    uint32_t vendorExtensionIndex,
                                                    const char** extensionName) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====SL_ANeuralNetworksDevice_getVendorExtensionName is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

int SL_ANeuralNetworksDevice_forEachVendorExtensionOperandTypeInformation(
        const ANeuralNetworksDevice* device, uint32_t vendorExtensionIndex, void* context,
        void (*callback)(SL_ANeuralNetworksExtensionOperandTypeInformation, void*)) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "=====SL_ANeuralNetworksDevice_"
                        "forEachVendorExtensionOperandTypeInformation is called ");

    return ANEURALNETWORKS_NO_ERROR;
}

#define NNCL_FUNC(symbol) .symbol = symbol

NnApiSLDriverImplFL5 slDriverImpl{
        .base{.implFeatureLevel = ANEURALNETWORKS_FEATURE_LEVEL_5},
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
    __android_log_print(ANDROID_LOG_VERBOSE, TAG_NAME,
                        "======ANeuralNetworks_getSLDriverImpl is called !!======");
    return reinterpret_cast<NnApiSLDriverImpl*>(&slDriverImpl);
}
}
