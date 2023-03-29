#pragma once

#include <tuple>
#include <type_traits>

#include "slang/type_system.h"

namespace slang {
namespace keyword {

template <typename T /*field type*/,
          typename =
              typename std::enable_if<type::is_tensor<typename T::type>::value>::type>
struct scale_of {
  using rtype = float;

  template <typename S /*signature type*/>
  rtype operator()(const S& sobject /*instance of signature*/) {
    auto field_tuple = sobject.field_tuple;
    auto field = std::get<T>(field_tuple);
    return field.scale();
  }
};

struct expr_placeholder
{
  using rtype = bool;

  template <typename S>
  bool operator()(const S& sobj) { return true; }
};

template <typename L, typename R, typename return_type>
struct expr_multiply {
  using rtype = return_type;

  expr_multiply() : lhs(L()), rhs(R()) {}
  expr_multiply(L& l, R& r) : lhs(l), rhs(r) {}

  template <typename S>
  return_type operator()(const S& sobj) {
    return lhs(sobj) * rhs(sobj);
  }

  L lhs;
  R rhs;
};

template <typename L, typename R>
struct expr_equal {
  using rtype = bool;

  expr_equal() : lhs(L()), rhs(R()) {}
  expr_equal(L& l, R& r) : lhs(l), rhs(r) {}

  template <typename S>
  bool operator()(const S& sobj) {
    return lhs(sobj) == rhs(sobj);
  }

  L lhs;
  R rhs;
};

template <
    typename T0,
    typename T1,
    typename = typename std::enable_if<
        std::is_same<typename T0::rtype, typename T1::rtype>::value>::type>
expr_multiply<T0, T1, float> operator*(T0 lhs, T1 rhs) {
  expr_multiply<T0, T1, float> expr_obj(lhs, rhs);
  return expr_obj;
}

template <typename T0, typename T1>
expr_equal<T0, T1> operator==(T0 lhs, T1 rhs) {
  expr_equal<T0, T1> expr_obj(lhs, rhs);
  return expr_obj;
}

}  // namespace keyword
}  // namespace slang
