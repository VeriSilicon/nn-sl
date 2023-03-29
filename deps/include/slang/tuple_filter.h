#pragma once
#include <type_traits>
#include "slang/type_system.h"
#include "slang/functional.h"
#include "slang/keyword.h"
namespace slang {

template <bool conditon, typename T /*rule type*/>
struct select_rule {
  using type = typename slang::type::rule<slang::keyword::expr_placeholder>;
};

template <typename T>
struct select_rule<true, T> {
  using type = T;
};

template <typename... Ts>
struct rule_filter {
  using type = std::tuple<Ts...>;
};

template <bool conditon, typename S /*signature type*/>
struct select_signature {
  using type = slang::type::none_type;
};

template <typename S>
struct select_signature<true, S> {
  using type = S;
};

template <typename... Ts>
struct signature_filter {
  using type = std::tuple<Ts...>;
};

template <typename S, typename T>
struct signature_table_instance_impl {
  void operator()(std::vector<S>& sig_table) {
    static_assert(std::is_same<S, decltype(T::instance())>::value);
    sig_table.push_back(T::instance());
  }
};

template <typename S>
struct signature_table_instance_impl<S, slang::type::none_type> {
  void operator()(std::vector<S>& sig_table) {}
};

template <size_t I, typename S, typename T>
struct signature_table_instance {
  void operator()(std::vector<S>& sig_table) {
    using stype = typename std::tuple_element<I, T>::type;

    signature_table_instance_impl<S, stype>()(sig_table);
    signature_table_instance<I - 1, S, T>()(sig_table);
  }
};

template <typename S, typename T>
struct signature_table_instance<0, S, T> {
  void operator()(std::vector<S>& sig_table) {
    using stype = typename std::tuple_element<0, T>::type;

    signature_table_instance_impl<S, stype>()(sig_table);
  }
};

template <typename S /*signature type*/, typename T /*filtered tuple*/>
struct signature_table {
  static std::vector<S> instance() {
    constexpr size_t sz = std::tuple_size<T>::value;
    std::vector<S> sig_table;
    signature_table_instance<sz - 1, S, T>()(sig_table);
    return sig_table;
  }
};

}  // namespace slang