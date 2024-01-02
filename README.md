# VSI NPU Android Support Library

**NOTE**: For customer, please ignore any section with (VSI internal)

## How to build from distributed customer source package

```sh
cmake -B <build_dir> -S <SL_dir> -DCMAKE_TOOLCHAIN_FILE=<ndk_root>/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34

cd <build_dir>
make tim-vx VsiSupportLibrary
# tim-vx MUST make before VsiSupportLibrary
```

## How to build from internal repo (VSI internal)

```sh
#Verified with android ndk r23c for verisilicon in house development
cmake -B <build_dir> -S <SL_dir> -DCMAKE_TOOLCHAIN_FILE=<ndk_root>/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DSLANG_TARGET_PID=<PID> -DSL_DIST_BUILD=OFF

cd <build_dir>
make tim-vx Slang VsiSupportLibrary
```

Reference for cmake variable in android toolchain: <https://developer.android.com/ndk/guides/cmake>

## Common problems

ld: error: undefined symbol: __android_log_print.

Append -DCMAKE_CXX_FLAGS="-llog" after cmake options may help.

## Switch git url for TIM-VX (VSI internal)

For customer, they can get latest tim-vx from github and this is the default behavior. For internal development, we can switch it back to internal gitlab by
    -DPUBLIC_TIM_VX=OFF

## Switch git url for Slang (internal build)

When build in internal, we can switch Slang resuorce to internal url by
    -DINTERNAL_BUILD=ON

## Integrate with Android

verified with i.MX 8M Plus and Android 14

### Precondition

Since Android Support Library is a standalone library which implemented NNAPI spec, on android, we need to
wrap it as a service for applications(android cts). In this document, we take "shell" approach to wrap support library as a service.

Download android aosp or get it from SoC vendor.

### Apply patch when build shell service

Apply patches in our SL `patches/`, if Android 12, use `patches_a12`:

```sh
cd ${AOSP_ROOT}/packages/modules/NeuralNetworks/
patch -p1 < 0001-Build-shell-service.patch
patch -p1 < 0002-Validate-model-in-shim-driver.patch
```

Why these patches are needed:

1. Build the shell service executable that can load our support library.
2. Use NNAPI validation utils to check whether a HAL model is conformed to NNAPI standard before converting the HAL model to SL model. Also check whether the HAL model contains OPs not supported by SL, if so, skip related VTS test cases.

### build shell service for VTS and CTS

```sh
cd ${AOSP_ROOT}/packages/modules/NeuralNetworks/driver/sample_shim
mm -j8
```

The built shell service executable is located at `${AOSP_ROOT}/out/target/product/evk_8mp/symbols/vendor/bin/hw/android.hardware.neuralnetworks-shell-service-sample`.

### Run test

push libtim-vx.so libVsiSupportLibrary.so libneuralnetworks.so VtsHalNeuralnetworksTargetTest CtsNNAPITestCases64 android.hardware.neuralnetworks-shell-service-sample to board
You can get android test suit in <https://source.android.com/>

#### 1. Delete old service and add shell service to vintf manifest

delete old service:
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

#### 2. Add system lib in default link space

to solve link fail in namespace(default): dlopen failed: library "libandroidfw.so" not found: needed by /vendor/lib64/libandroid.so in namespace (default)

In `linkerconfig/ld.config.txt`

```sh
[vendor]
namespace.default.search.paths = /odm/${LIB}
# Add these two lines
namespace.default.search.paths += /system/${LIB}
namespace.default.search.paths += /apex/com.android.i18n/${LIB}
namespace.default.search.paths += /apex/com.android.os.statsd/${LIB}
```

### 3. Start service by run android.hardware.neuralnetworks-shell-service-sample on Android board

### 4. run test with VTS

```sh
./VtsHalNeuralnetworksTargetTest --gtest_filter=TestGenerated/GeneratedTest.Test/android_hardware_neuralnetworks_IDevice_nnapi_sample_sl_updatable_reshape
```

### 5. run test with CTS

```sh
./CtsNNAPITestCases64 --gtest_filter=TestGenerated/QuantizationCouplingTest*
```

## Integrate with TfLite

### Get the source code of TensorFlow

```sh
git clone https://github.com/tensorflow/tensorflow.git
```

### Build benchmark_model

```sh
cd tensorflow/tensorflow/lite
git checkout v2.13.0
mkdir cmake_build && cd cmake_build
cmake .. -DCMAKE_TOOLCHAIN_FILE=<ndk_root>/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a
make benchmark_model -j8
```

push benchmark_model to board

### Run benchmark_model with support library

```sh
./benchmark_model --graph=mobilenet_v1_1.0_224_quant.tflite --use_nnapi=true --nnapi_support_library_path=/vendor/lib64/libVsiSupportLibrary.so --nnapi_accelerator_name=vsi-device-0
```

## How to pack source for release (VSI internal)

cd build directory
run `make tim-vx Slang VsiSupportLibrary && make package_source`, you will get source code in archived files.

Note: don't create build folder in your source code folder, else the package will include the build directory
