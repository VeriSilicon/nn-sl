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
#ifndef VSI_ANDROID_SL_TYPE_H
#define VSI_ANDROID_SL_TYPE_H

#include <vector>

#include <android/NeuralNetworksTypes.h>
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"

namespace vsi {
namespace android {
namespace sl {

enum class IOType { INPUT, OUTPUT };
struct Operand {
    Operand(ANeuralNetworksOperandType type) : type_info(type) {}

    ANeuralNetworksOperandType type_info;

    // symmetric per-channel quantized parameters
    std::vector<float> scales;
    uint32_t channel_dim;

    bool is_small_value{false};
    // store small value by copy
    std::vector<uint8_t> small_value;
    // sotre large value by reference
    const void* buffer{nullptr};
    size_t length{0};
};

struct Operation {
    ANeuralNetworksOperationType type;
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> outputs;
};

enum class MemoryType { FD, DESC, AHB };

/**
 * Operand types.
 *
 * The type of an operand in a model.
 *
 * Types prefaced with TENSOR_* must be used for tensor data (i.e., tensors
 * with at least one dimension). Types not prefaced by TENSOR_* represent
 * scalar values and must have no dimensions.
 *
 * Although we define many types, most operators accept just a few
 * types. Most used are {@link OperandType::TENSOR_FLOAT32},
 * {@link OperandType::TENSOR_QUANT8_ASYMM},
 * and {@link OperandType::INT32}.
 */
enum class OperandType {
    /** A 32 bit floating point scalar value. */
    FLOAT32 = 0,

    /** A signed 32 bit integer scalar value. */
    INT32 = 1,

    /** An unsigned 32 bit integer scalar value. */
    UINT32 = 2,

    /** A tensor of 32 bit floating point values. */
    TENSOR_FLOAT32 = 3,

    /** A tensor of 32 bit integer values. */
    TENSOR_INT32 = 4,

    /**
     * A tensor of 8 bit unsigned integers that represent real numbers.
     *
     * Attached to this tensor are two numbers that can be used to convert the
     * 8 bit integer to the real value and vice versa. These two numbers are:
     * - scale: a 32 bit floating point value greater than zero.
     * - zeroPoint: a 32 bit integer, in range [0, 255].
     *
     * The formula is:
     *   real_value = (integer_value - zeroPoint) * scale.
     */
    TENSOR_QUANT8_ASYMM = 5,

    /**
     * An 8 bit boolean scalar value.
     *
     * Values of this operand type are either true or false. A zero value
     * represents false; any other value represents true.
     */
    BOOL = 6,

    /**
     * A tensor of 16 bit signed integers that represent real numbers.
     *
     * Attached to this tensor is a number representing real value scale that is
     * used to convert the 16 bit number to a real value in the following way:
     * realValue = integerValue * scale.
     *
     * scale is a 32 bit floating point with value greater than zero.
     */
    TENSOR_QUANT16_SYMM = 7,

    /**
     * A tensor of IEEE 754 16 bit floating point values.
     */
    TENSOR_FLOAT16 = 8,

    /**
     * A tensor of 8 bit boolean values.
     *
     * Values of this operand type are either true or false. A zero value
     * represents false; any other value represents true.
     */
    TENSOR_BOOL8 = 9,

    /**
     * An IEEE 754 16 bit floating point scalar value.
     */
    FLOAT16 = 10,

    /**
     * A tensor of 8 bit signed integers that represent real numbers.
     *
     * This tensor is associated with additional fields that can
     * be used to convert the 8 bit signed integer to the real value and vice versa.
     * These fields are:
     * - channelDim: a 32 bit unsigned integer indicating channel dimension.
     * - scales: an array of positive 32 bit floating point values.
     * The size of the scales array must be equal to dimensions[channelDim].
     *
     * {@link SymmPerChannelQuantParams} must hold the parameters for an Operand of this type.
     * The channel dimension of this tensor must not be unknown (dimensions[channelDim] != 0).
     *
     * The formula is:
     * realValue[..., C, ...] =
     *     integerValue[..., C, ...] * scales[C]
     * where C is an index in the Channel dimension.
     */
    TENSOR_QUANT8_SYMM_PER_CHANNEL = 11,

    /**
     * A tensor of 16 bit unsigned integers that represent real numbers.
     *
     * Attached to this tensor are two numbers that can be used to convert the
     * 16 bit integer to the real value and vice versa. These two numbers are:
     * - scale: a 32 bit floating point value greater than zero.
     * - zeroPoint: a 32 bit integer, in range [0, 65535].
     *
     * The formula is:
     * real_value = (integer_value - zeroPoint) * scale.
     */
    TENSOR_QUANT16_ASYMM = 12,

    /**
     * A tensor of 8 bit signed integers that represent real numbers.
     *
     * Attached to this tensor is a number representing real value scale that is
     * used to convert the 8 bit number to a real value in the following way:
     * realValue = integerValue * scale.
     *
     * scale is a 32 bit floating point with value greater than zero.
     */
    TENSOR_QUANT8_SYMM = 13,

    /**
     * A tensor of 8 bit signed integers that represent real numbers.
     *
     * Attached to this tensor are two numbers that can be used to convert the
     * 8 bit integer to the real value and vice versa. These two numbers are:
     * - scale: a 32 bit floating point value greater than zero.
     * - zeroPoint: a 32 bit integer, in range [-128, 127].
     *
     * The formula is:
     * real_value = (integer_value - zeroPoint) * scale.
     */
    TENSOR_QUANT8_ASYMM_SIGNED = 14,

    /**
     * A reference to a subgraph.
     *
     * Must have the lifetime {@link Operand::LifeTime::SUBGRAPH}.
     */
    SUBGRAPH = 15,

    /**
     * DEPRECATED. Since HAL version 1.2, extensions are the preferred
     * alternative to OEM operation and data types.
     *
     * OEM specific scalar value.
     */
    OEM = 10000,

    /**
     * DEPRECATED. Since HAL version 1.2, extensions are the preferred
     * alternative to OEM operation and data types.
     *
     * A tensor of OEM specific values.
     */
    TENSOR_OEM_BYTE = 10001,
};

/**
 * The capabilities of a driver.
 *
 * This represents performance of non-extension operations.
 */
struct Capabilities {
    /**
     * Performance information for the reference workload.
     *
     * Used by a driver to report its performance characteristics.
     */
    struct PerformanceInfo {
        /**
         * Ratio of the time taken by the driver to execute the
         * workload compared to the time the CPU would take for the
         * same workload. A lower number is better.
         */
        float execTime = 0;

        /**
         * Ratio of the energy used by the driver compared to what
         * the CPU would use for doing the same workload. A lower number
         * is better.
         */
        float powerUsage = 0;
    };

    /**
     * Driver performance when operating on a particular data type.
     * In the case of float32 data, this is used when the calculations
     * are not relaxed.
     */
    struct OperandPerformance {
        OperandType type{};
        PerformanceInfo info;
    };

    class OperandPerformanceTable {
       public:
    //     static Result<OperandPerformanceTable> create(
    //             std::vector<OperandPerformance> operandPerformances);

    //     PerformanceInfo lookup(OperandType type) const;
    //     const std::vector<OperandPerformance>& asVector() const;

    //    private:
        explicit OperandPerformanceTable(std::vector<OperandPerformance> operandPerformances)
            : Sorted(std::move(operandPerformances)) {}
        std::vector<OperandPerformance> Sorted;
    };

    /**
     * Driver performance when operating on float32 data but performing
     * calculations with range and/or precision as low as that of the IEEE
     * 754 16-bit floating-point format.
     */
    PerformanceInfo relaxedFloat32toFloat16PerformanceScalar;
    PerformanceInfo relaxedFloat32toFloat16PerformanceTensor;

    /**
     * Performance by operand type. Must be sorted by OperandType.
     *
     * If a particular {@link OperandType} is not present in operandPerformance,
     * its performance is treated as
     * { .execTime = FLT_MAX, .powerUsage = FLT_MAX }.
     *
     * Performance does not apply to {@link OperandType::SUBGRAPH}, and a driver
     * must not report operand performance for {@link OperandType::SUBGRAPH}.
     */
    OperandPerformanceTable operandPerformance;

    /**
     * Performance of an {@link OperationType::IF} operation is the sum of
     * {@link Capabilities::ifPerformance} and the mean of performance for the
     * two branch subgraphs, where performance for a subgraph is the sum of the
     * performance of all operations within the subgraph.
     */
    PerformanceInfo ifPerformance;

    /**
     * Performance of a {@link OperationType::WHILE} operation is the sum of
     * {@link Capabilities::whilePerformance}, performance for the condition
     * subgraph and performance for the body subgraph, where performance for a
     * subgraph is the sum of the performance of all operations within the
     * subgraph.
     */
    PerformanceInfo whilePerformance;
};

}  // namespace sl
}  // namespace android
}  // namespace vsi

#endif