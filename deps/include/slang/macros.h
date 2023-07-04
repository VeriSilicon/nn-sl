#pragma once
#include "slang/functional.h"
#include "slang/keyword.h"
#include "slang/type_system.h"
#include "slang/tuple_filter.h"

#define BEGIN_SPEC(sname) \
  namespace op {          \
  namespace sname {

#define DECLARE_TENSOR_PARAM(pname, mod) \
  static constexpr char const * Role##pname##_asStr = #pname;          \
  struct Role##pname {                                      \
    static constexpr char const * doc = Role##pname##_asStr;   \
  };                                                        \
  using pname =                                             \
      slang::type::tensor_field<Role##pname, slang::type::modifier::k##mod>;

#define DECLARE_SCALAR_PARAM(pname, mod) \
  struct Role##pname;                    \
  using pname =                          \
      slang::type::scalar_field<Role##pname, slang::type::modifier::k##mod>;

#define DECLARE_SIGNATURE(...)                           \
  using signature = slang::type::signature<__VA_ARGS__>;

#define START_DEFINE_SIGNATURES()                                \
  constexpr auto TF32 = slang::type::data_type::kTF32;           \
  constexpr auto FP32 = slang::type::data_type::kFP32;           \
  constexpr auto FP16 = slang::type::data_type::kFP16;           \
  constexpr auto BF16 = slang::type::data_type::kBF16;           \
  constexpr auto INT64 = slang::type::data_type::kINT64;         \
  constexpr auto INT32 = slang::type::data_type::kINT32;         \
  constexpr auto UINT32 = slang::type::data_type::kUINT32;       \
  constexpr auto INT16 = slang::type::data_type::kINT16;         \
  constexpr auto UINT16 = slang::type::data_type::kUINT16;       \
  constexpr auto INT8 = slang::type::data_type::kINT8;           \
  constexpr auto UINT8 = slang::type::data_type::kUINT8;         \
  constexpr auto BOOL8 = slang::type::data_type::kBOOL8;         \
  constexpr auto INT4 = slang::type::data_type::kINT4;           \
  constexpr auto UINT4 = slang::type::data_type::kUINT4;         \
  constexpr auto NO_QUANT = slang::type::quant_type::kNONE;      \
  constexpr auto ASYMM = slang::type::quant_type::kASYMM;        \
  constexpr auto SYMM = slang::type::quant_type::kSYMM;          \
  constexpr auto SYMM_PCQ = slang::type::quant_type::kSYMM_PCQ;  \
  constexpr auto DFP = slang::type::quant_type::kDFP;            \
  constexpr auto CONSTANT = slang::type::tensor_attr::kCONSTANT; \
  constexpr auto VARIABLE = slang::type::tensor_attr::kVARIABLE;

#define TENSOR(...) slang::functional::tensor_wrapper<__VA_ARGS__>
#define SCALAR(...) slang::functional::scalar_wrapper<__VA_ARGS__>

#define DEFINE_SIGNATURE(name, ...) \
  using name = slang::functional::signature_wrapper<__VA_ARGS__>;

#define ADD_SIGNATURE(...)                                             \
  using sig_table_type = std::tuple<__VA_ARGS__>;                      \
  using sig_table = slang::signature_table<signature, sig_table_type>; \
  static std::vector<signature> signature_table = sig_table::instance();

#define START_DEFINE_RULES() using namespace slang::keyword;

#define DEFINE_RULE(name, ...) \
  using name = slang::type::rule<decltype(__VA_ARGS__)>;

#define ADD_RULE(...) static std::tuple<__VA_ARGS__> rule_tuple;

#define END_SPEC(sname)                                                        \
  }                                                                            \
  }                                                                            \
  template <>                                                                  \
  bool slang::functional::check_signature_impl<op::sname::signature>(          \
      const std::vector<op::sname::signature>& sig_table,                      \
      op::sname::signature& sObject) {                                         \
    bool available = true;                                                     \
    for (auto& sig : sig_table) {                                              \
      auto sig_size = std::tuple_size<decltype(sig.field_tuple)>::value;       \
      if (sig_size != std::tuple_size<decltype(sObject.field_tuple)>::value) { \
        available = false;                                                     \
        continue;                                                              \
      }                                                                        \
      available = slang::functional::compare_field_tuple(sig.field_tuple,      \
                                                         sObject.field_tuple); \
      if (available) break;                                                    \
    }                                                                          \
    return available;                                                          \
  }                                                                            \
  template <>                                                                  \
  bool slang::functional::check_signature<op::sname::signature>(               \
      op::sname::signature & sObject) {                                        \
    return check_signature_impl(op::sname::signature_table, sObject);          \
  }                                                                            \
  template <>                                                                  \
  bool slang::functional::check_rule<op::sname::signature>(                    \
      op::sname::signature & sObject) {                                        \
    return slang::functional::check_rule_impl(op::sname::rule_tuple, sObject); \
  }


