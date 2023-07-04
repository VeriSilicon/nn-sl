# Build

Please build OpenVX driver for Android platform firstly, and install the libraries into prebuilt/android_arm64.
Please check the directory structure aligned with prebuilt/android_arm64/install_sdk.txt

```
# change ABI if required
cmake -B <build_dir> -S <SL_dir> -DCMAKE_TOOLCHAIN_FILE=<ndk_root>/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a

cd <build_dir>
make tim-vx VsiSupportLibrary
# tim-vx MUST make before VsiSupportLibrary
```

Reference for cmake variable in android toolchain: https://developer.android.com/ndk/guides/cmake


# Integrate with Android

## Precondition

Since Android Support Library is a standalone library which implemented NNAPI spec, on android, we need to
wrap it as a service for applications(android cts). In this document, we take "shell" approach to wrap support library
as a service.

Download android AOSP or get it from SoC vendor.

## Build Shell Service
```sh
cd <aosp_root_dir>/packages/modules/NeuralNetworks/driver/sample_shim
patch -p1 < shell-service.patch # you can get the patch file in our package
mm -j8
# service bianry name is android.hardware.neuralnetworks-shell-service-sample
```

## Compile VTS for verification
```sh
cd <aosp_root_dir>/packages/hardware/interfaces/neuralnetworks/aidl/vts/functional
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