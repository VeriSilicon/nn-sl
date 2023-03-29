#pragma once
#include "type_system.h"

namespace slang {

namespace functional {

template <typename T,
          slang::type::data_type D,
          slang::type::quant_type Q,
          slang::type::tensor_attr A = slang::type::tensor_attr::kVARIABLE>
struct tensor_wrapper {
  static constexpr slang::type::data_type dtype = D;
  static constexpr slang::type::quant_type qtype = Q;
  static constexpr slang::type::tensor_attr atype = A;

  using type = T;

  static T instance() { return T(dtype, qtype, atype); }
};

template <typename T, slang::type::data_type D>
struct scalar_wrapper {
  static constexpr slang::type::data_type dtype = D;

  using type = T;

  static T instance() { return T(dtype); }
};

namespace tuple {

template <typename L, typename... T>
struct tuple_push_back {};

template <template <typename...> typename L, typename... U, typename... T>
struct tuple_push_back<L<U...>, T...> {
  using type = L<U..., T...>;
};

template <typename L, typename... T>
struct tuple_push_front {};

template <template <typename...> typename L, typename... U, typename... T>
struct tuple_push_front<L<U...>, T...> {
  using type = L<T..., U...>;
};

template <size_t sz,
          typename F /*field_instance_wrapper*/,
          typename S /*source tuple*/>
struct tuple_instance {
  template <typename T>
  void operator()(T& target) {
    using executor = typename F::template field_instance<sz, S>;
    executor()(target);
    tuple_instance<sz - 1, F, S>()(target);
  }
};

template <typename F, typename S>
struct tuple_instance<0, F, S> {
  template <typename T>
  void operator()(T& target) {
    using executor = typename F::template field_instance<0, S>;
    executor()(target);
  }
};

}  // namespace tuple

template <size_t sz, typename W0, typename... W>
struct unwrap {
  using stype =
      typename tuple::tuple_push_front<typename unwrap<sz - 1, W...>::stype,
                                    typename W0::type>::type;
};
template <typename W>
struct unwrap<1, W> {
  using stype = ::slang::type::signature<typename W::type>;
};

struct field_instance_wrapper {
  template <size_t index, typename S /*source tuple*/>
  struct field_instance {
    template <typename T>
    void operator()(T& target) {
      std::get<index>(target) = std::tuple_element<index, S>::type::instance();
    }
  };
};

template <typename... W /*field wrapper*/>
struct signature_wrapper {
  using signature_type = typename unwrap<sizeof...(W), W...>::stype;
  using wrapped_field_tuple_t = std::tuple<W...>;
  using field_tuple_t = typename signature_type::type;

  static signature_type instance() {
    constexpr std::size_t sz = sizeof...(W);
    field_tuple_t sig_instance;
    using executor = tuple::
        tuple_instance<sz - 1, field_instance_wrapper, wrapped_field_tuple_t>;
    executor()(sig_instance);
    return signature_type(sig_instance);
  }
};

template <typename T>
typename std::enable_if<std::is_same<typename T::type, type::tensor_tag>::value,
                        bool>::type
compare_field(T legal_field, T instance_field) {
  if (legal_field.storage.dtype != instance_field.storage.dtype ||
      legal_field.storage.qtype != instance_field.storage.qtype ||
      legal_field.storage.attr != instance_field.storage.attr) {
    return false;
  }
  return true;
}

template <typename T>
typename std::enable_if<std::is_same<typename T::type, type::scalar_tag>::value,
                        bool>::type
compare_field(T legal_field, T instance_field) {
  if (legal_field.storage.dtype != instance_field.storage.dtype) {
    return false;
  }
  return true;
}

template <size_t I = 0, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), bool>::type compare_field_tuple(
    std::tuple<Ts...> legal_fields, std::tuple<Ts...> instance_fields) {
  return true;
}

template <size_t I = 0, typename... Ts>
    typename std::enable_if <
    I<sizeof...(Ts), bool>::type compare_field_tuple(
        std::tuple<Ts...> legal_fields, std::tuple<Ts...> instance_fields) {
  auto legal_field = std::get<I>(legal_fields);
  auto instance_field = std::get<I>(instance_fields);
  if (legal_field.m == type::modifier::kOPTIONAL &&
      instance_field.storage.dtype == type::data_type::kINVALID) {
    return compare_field_tuple<I + 1>(legal_fields, instance_fields);
  } else {
    if (typeid(legal_field) != typeid(instance_field)) {
      return false;
    } else {
      if (!compare_field(legal_field, instance_field)) return false;
    }
  }

  return compare_field_tuple<I + 1>(legal_fields, instance_fields);
}

template <typename S /*operator signature type*/>
static bool check_signature_impl(const std::vector<S>& sig_table, S& sig_inst) {
  std::cout << "Unknown signature type" << std::endl;
  return false;
}

template <typename S>
static bool check_signature(S& obj) {
  std::cout << "Unknown signature type" << std::endl;
  return false;
}

template <typename T>
slang::type::rule<T> make_rule(T expr) {
  return slang::type::rule<T>(expr);
}

template <size_t I = 0, typename... Ts, typename S>
typename std::enable_if<I == sizeof...(Ts), bool>::type compare_rule(
    std::tuple<Ts...> rule_tuple, S signature) {
  return true;
}

template <size_t I = 0, typename... Ts, typename S>
    typename std::enable_if <
    I<sizeof...(Ts), bool>::type compare_rule(std::tuple<Ts...> rule_tuple,
                                              S signature) {
  if (!std::get<I>(rule_tuple)(signature)) return false;

  return compare_rule<I + 1>(rule_tuple, signature);
}

template <typename T /*rule tuple*/, typename S /*operator signature type*/>
bool check_rule_impl(const T& rule_tuple, const S& sobj) {
  return slang::functional::compare_rule(rule_tuple, sobj);
}

template <typename S /*operator signature type*/>
static bool check_rule(S& sobj) {
  std::cout << "Unknown signature type" << std::endl;
  return false;
}

}  // namespace functional
}  // namespace slang
