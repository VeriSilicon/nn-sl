
#ifndef __PID_NAME__
#error "PID Name Not given"
#endif

#define CAT_TOKEN_1(A, B) A##B
#define CAT_TOKEN(A, B) CAT_TOKEN_1(A##_, B)

#define __PID_SPEC_DEF_OP(OP_NAME, PNAME)                \
  using CAT_TOKEN(OP_NAME, PNAME) =                      \
      ::hw::spec::op_signature<::op::OP_NAME::signature, \
                               ::hw::spec::feature_bits<PNAME>>

#include "spec/hw/activation.spec.h"
#include "spec/hw/batch_to_space.spec.h"
#include "spec/hw/conv2d.spec.h"
#include "spec/hw/depthwise_conv2d.spec.h"
#include "spec/hw/depth_to_space.spec.h"
#include "spec/hw/dequantize.spec.h"
#include "spec/hw/eltwise.spec.h"
#include "spec/hw/eltwise_unary.spec.h"
#include "spec/hw/floor.spec.h"
#include "spec/hw/fully_connected.spec.h"
#include "spec/hw/grouped_conv2d.spec.h"
#include "spec/hw/l2_normalization.spec.h"
#include "spec/hw/logical_not.spec.h"
#include "spec/hw/logical_and_or.spec.h"
#include "spec/hw/mean.spec.h"
#include "spec/hw/pad.spec.h"
#include "spec/hw/pad_v2.spec.h"
#include "spec/hw/pool2d.spec.h"
#include "spec/hw/pow.spec.h"
#include "spec/hw/prelu.spec.h"
#include "spec/hw/quantize.spec.h"
#include "spec/hw/reduce_all_any.spec.h"
#include "spec/hw/reduce_max_min_prod_sum.spec.h"
#include "spec/hw/relational_op.spec.h"
#include "spec/hw/reshape.spec.h"
#include "spec/hw/rsqrt.spec.h"
#include "spec/hw/softmax.spec.h"
#include "spec/hw/space_to_depth.spec.h"
#include "spec/hw/space_to_batch.spec.h"
#include "spec/hw/transpose_conv2d.spec.h"
__PID_SPEC_DEF_OP(activation, __PID_NAME__);
__PID_SPEC_DEF_OP(batch_to_space, __PID_NAME__);
__PID_SPEC_DEF_OP(conv2d, __PID_NAME__);
__PID_SPEC_DEF_OP(depthwise_conv2d, __PID_NAME__);
__PID_SPEC_DEF_OP(depth_to_space, __PID_NAME__);
__PID_SPEC_DEF_OP(dequantize, __PID_NAME__);
__PID_SPEC_DEF_OP(eltwise, __PID_NAME__);
__PID_SPEC_DEF_OP(eltwise_unary, __PID_NAME__);
__PID_SPEC_DEF_OP(floor, __PID_NAME__);
__PID_SPEC_DEF_OP(fully_connected, __PID_NAME__);
__PID_SPEC_DEF_OP(grouped_conv2d, __PID_NAME__);
__PID_SPEC_DEF_OP(l2_normalization, __PID_NAME__);
__PID_SPEC_DEF_OP(logical_not, __PID_NAME__);
__PID_SPEC_DEF_OP(logical_and_or, __PID_NAME__);
__PID_SPEC_DEF_OP(mean, __PID_NAME__);
__PID_SPEC_DEF_OP(pad, __PID_NAME__);
__PID_SPEC_DEF_OP(pad_v2, __PID_NAME__);
__PID_SPEC_DEF_OP(pool2d, __PID_NAME__);
__PID_SPEC_DEF_OP(pow, __PID_NAME__);
__PID_SPEC_DEF_OP(prelu, __PID_NAME__);
__PID_SPEC_DEF_OP(quantize, __PID_NAME__);
__PID_SPEC_DEF_OP(reduce_all_any, __PID_NAME__);
__PID_SPEC_DEF_OP(reduce_max_min_prod_sum, __PID_NAME__);
__PID_SPEC_DEF_OP(relational_op, __PID_NAME__);
__PID_SPEC_DEF_OP(reshape, __PID_NAME__);
__PID_SPEC_DEF_OP(rsqrt, __PID_NAME__);
__PID_SPEC_DEF_OP(softmax, __PID_NAME__);
__PID_SPEC_DEF_OP(space_to_depth, __PID_NAME__);
__PID_SPEC_DEF_OP(space_to_batch, __PID_NAME__);
__PID_SPEC_DEF_OP(transpose_conv2d, __PID_NAME__);

// todo: table of operator
