## Standard: 73, implement: 60. Non-standard: 32, implement: 4

| NNAPI Runtime function                                                    | Status | feature level |
| -                                                                         | -      | -             |
| ANeuralNetworksBurst_create                                               | No     | 5             |
| ANeuralNetworksBurst_free                                                 | No     | 5             |
| ANeuralNetworksCompilation_createForDevices                               | Yes    | 5             |
| ANeuralNetworksCompilation_create                                         | Yes    | 5             |
| ANeuralNetworksCompilation_finish                                         | Yes    | 5             |
| ANeuralNetworksCompilation_free                                           | Yes    | 5             |
| ANeuralNetworksCompilation_getPreferredMemoryAlignmentForInput            | No     | 5             |
| ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput           | No     | 5             |
| ANeuralNetworksCompilation_getPreferredMemoryPaddingForInput              | No     | 5             |
| ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput             | No     | 5             |
| ANeuralNetworksCompilation_setCaching                                     | Yes    | 5             |
| ANeuralNetworksCompilation_setPreference                                  | Yes    | 5             |
| ANeuralNetworksCompilation_setPriority                                    | Yes    | 5             |
| ANeuralNetworksCompilation_setTimeout                                     | Yes    | 5             |
| ANeuralNetworksDevice_getExtensionSupport                                 | Won't Support | 5      |
| ANeuralNetworksDevice_getFeatureLevel                                     | Yes    | 5             |
| ANeuralNetworksDevice_getName                                             | Yes    | 5             |
| ANeuralNetworksDevice_getType                                             | Yes    | 5             |
| ANeuralNetworksDevice_getVersion                                          | Yes    | 5             |
| ANeuralNetworksDevice_wait                                                | No     | 5             |
| ANeuralNetworksEvent_createFromSyncFenceFd                                | Yes    | 5             |
| ANeuralNetworksEvent_free                                                 | Yes    | 5             |
| ANeuralNetworksEvent_getSyncFenceFd                                       | Yes    | 5             |
| ANeuralNetworksEvent_wait                                                 | Yes    | 5             |
| ANeuralNetworksExecution_burstCompute                                     | No     | 5             |
| ANeuralNetworksExecution_compute                                          | Yes    | 5             |
| ANeuralNetworksExecution_create                                           | Yes    | 5             |
| ANeuralNetworksExecution_enableInputAndOutputPadding                      | No     | 5             |
| ANeuralNetworksExecution_free                                             | Yes    | 5             |
| ANeuralNetworksExecution_getDuration                                      | Yes    | 5             |
| ANeuralNetworksExecution_getOutputOperandDimensions                       | Yes    | 5             |
| ANeuralNetworksExecution_getOutputOperandRank                             | Yes    | 5             |
| ANeuralNetworksExecution_setInput                                         | Yes    | 5             |
| ANeuralNetworksExecution_setInputFromMemory                               | Yes    | 5             |
| ANeuralNetworksExecution_setLoopTimeout                                   | Yes    | 5             |
| ANeuralNetworksExecution_setMeasureTiming                                 | Yes    | 5             |
| ANeuralNetworksExecution_setOutput                                        | Yes    | 5             |
| ANeuralNetworksExecution_setOutputFromMemory                              | Yes    | 5             |
| ANeuralNetworksExecution_setReusable                                      | Yes    | 5             |
| ANeuralNetworksExecution_setTimeout                                       | Yes    | 5             |
| ANeuralNetworksExecution_startCompute                                     | Yes    | 5             |
| ANeuralNetworksExecution_startComputeWithDependencies                     | Yes    | 5             |
| ANeuralNetworksMemoryDesc_addInputRole                                    | Yes    | 5             |
| ANeuralNetworksMemoryDesc_addOutputRole                                   | Yes    | 5             |
| ANeuralNetworksMemoryDesc_create                                          | Yes    | 5             |
| ANeuralNetworksMemoryDesc_finish                                          | Yes    | 5             |
| ANeuralNetworksMemoryDesc_free                                            | Yes    | 5             |
| ANeuralNetworksMemoryDesc_setDimensions                                   | Yes    | 5             |
| ANeuralNetworksMemory_copy                                                | Yes    | 5             |
| ANeuralNetworksMemory_createFromAHardwareBuffer                           | Yes    | 5             |
| ANeuralNetworksMemory_createFromDesc                                      | Yes    | 5             |
| ANeuralNetworksMemory_createFromFd                                        | Yes    | 5             |
| ANeuralNetworksMemory_free                                                | Yes    | 5             |
| ANeuralNetworksModel_addOperand                                           | Yes    | 5             |
| ANeuralNetworksModel_addOperation                                         | Yes    | 5             |
| ANeuralNetworksModel_create                                               | Yes    | 5             |
| ANeuralNetworksModel_finish                                               | Yes    | 5             |
| ANeuralNetworksModel_free                                                 | Yes    | 5             |
| ANeuralNetworksModel_getExtensionOperandType                              | Won't support | 5      |
| ANeuralNetworksModel_getExtensionOperationType                            | Won't Support | 5      |
| ANeuralNetworksModel_getSupportedOperationsForDevices                     | Yes    | 5             |
| ANeuralNetworksModel_identifyInputsAndOutputs                             | Yes    | 5             |
| ANeuralNetworksModel_relaxComputationFloat32toFloat16                     | Yes    | 5             |
| ANeuralNetworksModel_setOperandExtensionData                              | won't suppoprt | 5     |
| ANeuralNetworksModel_setOperandSymmPerChannelQuantParams                  | Yes    | 5             |
| ANeuralNetworksModel_setOperandValue                                      | Yes    | 5             |
| ANeuralNetworksModel_setOperandValueFromMemory                            | Yes    | 5             |
| ANeuralNetworksModel_setOperandValueFromModel                             | Yes    | 5             |
| ANeuralNetworks_getDefaultLoopTimeout                                     | Yes    | 5             |
| ANeuralNetworks_getDevice                                                 | Yes    | 5             |
| ANeuralNetworks_getDeviceCount                                            | Yes    | 5             |
| ANeuralNetworks_getMaximumLoopTimeout                                     | Yes    | 5             |
| ANeuralNetworks_getRuntimeFeatureLevel                                    | Yes    | 5             |

**NOTE**: test result with imx8mp android 14