
#ifndef __PID_NAME__
#error "PID Name Not given"
#endif

#define CAT_TOKEN_1(A, B) A##B
#define CAT_TOKEN(A, B) CAT_TOKEN_1(A##_, B)

#define __PID_SPEC_DEF_OP(OP_NAME, PNAME)                \
  using CAT_TOKEN(OP_NAME, PNAME) =                      \
      ::hw::spec::op_signature<::op::OP_NAME::signature, \
                               ::hw::spec::feature_bits<PNAME>>

#include "spec/hw/conv2d.spec.h"
#include "spec/hw/depthwise_conv2d.spec.h"
#include "spec/hw/average_pool.spec.h"
#include "spec/hw/reshape.spec.h"
#include "spec/hw/softmax.spec.h"
#include "spec/hw/pad.spec.h"
#include "spec/hw/pad_v2.spec.h"
__PID_SPEC_DEF_OP(conv2d, __PID_NAME__);

// todo: table of operator
