#pragma once

namespace hw {
namespace spec {
template <typename S /*signature type*/,
          typename T /*hardware feature bit triator*/>
struct op_signature {};
}  // namespace spec
}  // namespace hw

#define SELECT_SIGNATURE(feature, signature) \
  typename slang::select_signature<(feature), signature>::type

#define SELECT_RULE(feature, rule) \
  typename slang::select_rule<(feature), rule>::type

#define OP_SIGNATURE_BEGIN(opname)                  \
  namespace hw {                                    \
  namespace spec {                                  \
  template <typename T>                             \
  struct op_signature<::op::opname::signature, T> { \
    using sig_type = ::op::opname::signature;

#define MAKE_SIGNATURE_TABLE(...)                                             \
  using sig_table_type = typename slang::signature_filter<__VA_ARGS__>::type; \
  using sig_table = slang::signature_table<sig_type, sig_table_type>;

#define MAKE_RULE_TABLE(...) \
  using rule_table_type = typename slang::rule_filter<__VA_ARGS__>::type;

#define OP_SIGNATURE_END(opname)                                         \
  op_signature() { sig_tbl_ = sig_table::instance(); }                   \
  bool check_signature(::op::opname::signature& sObject) {               \
    return ::slang::functional::check_signature_impl(sig_tbl_, sObject); \
  }                                                                      \
  bool check_rule(::op::opname::signature& sObject) {                    \
    return ::slang::functional::check_rule_impl(rule_tbl_, sObject);     \
  }                                                                      \
  std::vector<sig_type> sig_tbl_;                                        \
  rule_table_type rule_tbl_;                                             \
  };                                                                     \
  }                                                                      \
  }
  