**NOTE**: For customer, please ignore any section with (VSI internal)
# How to build from distributed customer source package
```
cmake -B <build_dir> -S <SL_dir> -DCMAKE_TOOLCHAIN_FILE=<ndk_root>/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a

cd <build_dir>
make tim-vx VsiSupportLibrary
# tim-vx MUST make before VsiSupportLibrary
```
# How to build from internal repo(VSI internal)

```bash
#Verified with android ndk r23c for verisilicon in house development
cmake -B <build_dir> -S <SL_dir> -DCMAKE_TOOLCHAIN_FILE=<ndk_root>/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DSLANG_TARGET_PID=<PID> -DSL_DIST_BUILD=OFF

cd <build_dir>
make tim-vx Slang VsiSupportLibrary
```
Reference for cmake variable in android toolchain: https://developer.android.com/ndk/guides/cmake
# Common problem
ld: error: undefined symbol: __android_log_print.

Append -DCMAKE_CXX_FLAGS="-llog" after cmake options may help.

# Switch git url for TIM-VX (VSI internal)
For customer, they can get latest tim-vx from github and this is the default behavior. For internal development, we can switch it back to internal gitlab by
    -DPUBLIC_TIM_VX=OFF

# Integrate with Android
verified with i.MX 8M Plus and Android 12

## Precondition

Since Android Support Library is a standalone library which impelemented NNAPI spec, on android, we need to
wrap it as a service for applications(android cts). In this document, we take "shell" approach to wrap support library
as a service.

Download android aosp or get it from SoC vendor.

## Build Shell Service
```sh
cd <aosp_root_dir>/packages/modules/NeuralNetworks/driver/sample_shim
patch -p1 < shell-service.patch # you can get the patch file in our package
mm -j8
# service bianry name is android.hardware.neuralnetworks-shell-service-sample
```


## Compile VTS for verification
```sh
cd <aosp_root_dir>/hardware/interfaces/neuralnetworks/aidl/vts/functional
mm -j8
# test binary name is VtsHalNeuralnetworksTargetTest
```

## Run test

push libtim-vx.so libVsiSupportLibrary.so VtsHalNeuralnetworksTargetTest android.hardware.neuralnetworks-shell-service-sample to board
### 1. Add shell service to vintf manifest
add following content to `/vendor/etc/vintf/manifest.xml`
```
<hal format="aidl">
    <name>android.hardware.neuralnetworks</name>
    <fqname>IDevice/nnapi-sample_sl_updatable</fqname>
</hal>
```

### 2. Start service by run android.hardware.neuralnetworks-shell-service-sample on Android board
### 3. run test with
```sh
./VtsHalNeuralnetworksTargetTest --gtest_filter=TestGenerated/GeneratedTest.Test/android_hardware_neuralnetworks_IDevice_nnapi_sample_sl_updatable_reshape
```
# Integrate with TfLite
## Get the source code of TensorFlow
```sh
git clone https://github.com/tensorflow/tensorflow.git
```
## Build benchmark_model
```sh
cd tensorflow/tensorflow/lite
mkdir cmake_build && cd cmake_build
cmake .. -DCMAKE_TOOLCHAIN_FILE=<ndk_root>/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a
make benchmark_model -j8
```
push benchmark_model to board
## Run benchmark_model with support library
```sh
./benchmark_model --graph=mobilenet_v1_1.0_224_quant.tflite --use_nnapi=true --nnapi_support_library_path=/vendor/lib64/libVsiSupportLibrary.so --nnapi_accelerator_name=vsi-device-0
```
# How to pack source for release (VSI internal)
cd build directory
run `make tim-vx Slang VsiSupportLibrary && make package_source`, you will get source code in archived files.

Note: don't create build folder in your source code folder, else the package will include the build directory
