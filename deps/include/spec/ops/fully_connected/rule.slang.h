START_DEFINE_RULES()

DEFINE_RULE(rule_scale, scale_of<Input>() * scale_of<Weight>() == scale_of<Bias>())


#ifndef __ENABLE_HW_SPEC__

ADD_RULE()

#endif