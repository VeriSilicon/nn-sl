# VSI NPU Android Support Library

## 1 How to build

with prebuild sdk is not recommend, just for build test.

```sh
cmake -B <build_dir> -S <SL_dir> -DCMAKE_TOOLCHAIN_FILE=<ndk_root>/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34

cd <build_dir>
make tim-vx VsiSupportLibrary
```
### Build with specific OpenVX driver (recommended)

add cmake definition -DEXT_VIV_SDK=PATH_OPENVX_SDK where OPENVX_SDK is the OpenVX driver libraries built from source in AOSP. Usually, SoC vendor is the responsible for updating this SDK.

The OPENVX_SDK directory should originized as in prebuild/android_arm64/

Reference for cmake variable in android toolchain: <https://developer.android.com/ndk/guides/cmake>

### Build with specific TIM-VX version

Addition cmake definition added for this purpose -DTIM_VX_TAG=commit-sha-id, just provide the commit id in TIM-VX github repo.

## 2 Integrate with Android

verified with i.MX 8M Plus and Android 14

### 2.1 Precondition

Since Android Support Library is a standalone library which implemented NNAPI spec, on android, we need to
wrap it as a service for applications(android cts). In this document, we take "shell" approach to wrap support library as a service.

Download android aosp or get it from SoC vendor.

### 2.2 Apply patch when build shell service

Apply patches in our SL `patches/`, select the corresponding version:

```sh
cd ${AOSP_ROOT}/packages/modules/NeuralNetworks/
patch -p1 < 0001-Build-shell-service.patch
patch -p1 < 0002-Validate-model-in-shim-driver.patch
```

Why these patches are needed:

1. Build the shell service executable that can load our support library.
2. Use NNAPI validation utils to check whether a HAL model is conformed to NNAPI standard before converting the HAL model to SL model. Also check whether the HAL model contains OPs not supported by SL, if so, skip related VTS test cases.

### 2.3 build shell service for VTS and CTS

```sh
cd ${AOSP_ROOT}/packages/modules/NeuralNetworks/driver/sample_shim
mm -j8
```

The built shell service executable is located at `${AOSP_ROOT}/out/target/product/evk_8mp/symbols/vendor/bin/hw/android.hardware.neuralnetworks-shell-service-sample`.

### 2.4 Run test

push libtim-vx.so libVsiSupportLibrary.so VtsHalNeuralnetworksTargetTest CtsNNAPITestCases64 android.hardware.neuralnetworks-shell-service-sample to board
You can get android test suit in <https://source.android.com/>

#### 2.4.1 Delete old service(optional) and add shell service to vintf manifest

If you have old service, delete old service:
cd /vendor/etc/vintf/manifest
rm -f android.hardware.neuralnetworks@1.3-service-vsi-npu-server.xml

Note: This change only for CTS & VTS
add following content to `/vendor/etc/vintf/manifest.xml`

```xml
<hal format="aidl">
    <name>android.hardware.neuralnetworks</name>
    <fqname>IDevice/nnapi-sample_sl_updatable</fqname>
</hal>
```

Finally, reboot.

#### 2.4.2 Add system lib in default link space

to solve link fail in namespace(default): dlopen failed: library "libandroidfw.so" not found: needed by /vendor/lib64/libandroid.so in namespace (default)

In `linkerconfig/ld.config.txt`

```sh
[vendor]
namespace.default.search.paths = /odm/${LIB}
# Add these three lines
namespace.default.search.paths += /system/${LIB}
namespace.default.search.paths += /apex/com.android.i18n/${LIB}
namespace.default.search.paths += /apex/com.android.os.statsd/${LIB}
```

### 2.5 Start service by run android.hardware.neuralnetworks-shell-service-sample on Android board

### 2.6. run test with VTS

```sh
./VtsHalNeuralnetworksTargetTest --gtest_filter=TestGenerated/GeneratedTest.Test/android_hardware_neuralnetworks_IDevice_nnapi_sample_sl_updatable_reshape
```

### 2.7 run test with CTS

```sh
./CtsNNAPITestCases64 --gtest_filter=TestGenerated/QuantizationCouplingTest*
```

## 3 Integrate with TfLite

### 3.1 Get the source code of TensorFlow

```sh
git clone https://github.com/tensorflow/tensorflow.git
```

### 3.2 Build benchmark_model

```sh
cd tensorflow/tensorflow/lite
git checkout v2.13.0
mkdir cmake_build && cd cmake_build
cmake .. -DCMAKE_TOOLCHAIN_FILE=<ndk_root>/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a
make benchmark_model -j8
```

push benchmark_model to board

### 3.3 Run benchmark_model with support library

```sh
./benchmark_model --graph=mobilenet_v1_1.0_224_quant.tflite --use_nnapi=true --nnapi_support_library_path=/vendor/lib64/libVsiSupportLibrary.so --nnapi_accelerator_name=vsi-device-0
```