START_DEFINE_RULES()

DEFINE_RULE(rule_scale, scale_of<Input>() * scale_of<Kernel>() == scale_of<Bias>())
DEFINE_RULE(rule_scale2, scale_of<Bias>() == scale_of<Input>() * scale_of<Kernel>())

#ifndef __ENABLE_HW_SPEC__

ADD_RULE(rule_scale, rule_scale2)

#endif
