#ifndef _SLANG_TYPE_SYSTEM_H_
#define _SLANG_TYPE_SYSTEM_H_

#include <cstring>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

namespace slang {
namespace type {

enum class data_type {
  kTF32,     /** TensorFlow 32bit float for training*/
  kFP32,     /** float point 32bit */
  kFP16,     /** float point 16bit */
  kBF16,     /** bfloat 16*/

  kINT64,
  // kUINT64,

  kINT32,
  kUINT32,

  kINT16,
  kUINT16,

  kINT8,
  kUINT8,

  kBOOL8,

  kINT4,
  kUINT4,

  kCNT,
  kINVALID = kCNT
};

static std::string as_string(data_type dtype) {
  std::string str;
  switch (dtype) {
    case data_type::kFP32:
    case data_type::kTF32:
      str = "float32";
      break;
    case data_type::kFP16:
      str = "float16";
      break;
    case data_type::kBF16:
      str = "bfloat16";
      break;
    case data_type::kINT64:
      str = "int64";
      break;
    case data_type::kINT32:
      str = "int32";
      break;
    case data_type::kUINT32:
      str = "uint32";
      break;
    case data_type::kINT16:
      str = "int16";
      break;
    case data_type::kUINT16:
      str = "uint16";
      break;
    case data_type::kINT8:
      str = "int8";
      break;
    case data_type::kUINT8:
      str = "uint8";
      break;
    case data_type::kINT4:
      str = "int4";
      break;
    case data_type::kUINT4:
      str = "uint4";
      break;
    case data_type::kBOOL8:
      str = "bool";
      break;
    default:
      str = "invalid";
  }
  return str;
}

enum class pad_type {
  kNONE,
  kSAME,
  kVALID
};

enum class activation_type {
  kNONE,
  kRELU,
  kRELU1,
  kRELU6
};

enum class layout_type {
  kNHWC,
  kNCHW
};

// Function to iterate through all values
// I equals number of values in tuple
template <size_t I = 0,
          //   typename visitor,
          typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type Tuple_For(
    std::tuple<Ts...> tup) {
  // If iterated through all values
  // of tuple, then simply return.
  return;
}

template <size_t I = 0,
          //   typename visitor,
          typename... Ts>
typename std::enable_if<(I < sizeof...(Ts)), void>::type Tuple_For(
    std::tuple<Ts...> tup
    // , visitor
) {
  // Print element of tuple
  // visitor(std::get<I>(tup)) ;
  auto field_instance = std::get<I>(tup);
  field_instance.doc();

  // Go to next element
  Tuple_For<I + 1
            // , visitor
            >(tup);
}

// template <typename rule>
// struct rule_visitor {
//     void operator()(const rule& r) {
//         std::cout << "visit rule " << std::endl;
//     }
// };

template <typename CPP_TYPE>
struct value_type_convertor {
  static constexpr data_type value = data_type::kINVALID;
};

template <>
struct value_type_convertor<float> {
  static constexpr data_type value = data_type::kFP32;
};


template <>
struct value_type_convertor<int32_t> {
  static constexpr data_type value = data_type::kINT32;
};

template <>
struct value_type_convertor<uint32_t> {
  static constexpr data_type value = data_type::kUINT32;
};

template <>
struct value_type_convertor<int16_t> {
  static constexpr data_type value = data_type::kINT16;
};

template <>
struct value_type_convertor<uint16_t> {
  static constexpr data_type value = data_type::kUINT16;
};
template <>
struct value_type_convertor<int8_t> {
  static constexpr data_type value = data_type::kINT8;
};

template <>
struct value_type_convertor<uint8_t> {
  static constexpr data_type value = data_type::kUINT8;
};

template <>
struct value_type_convertor<bool> {
  static constexpr data_type value = data_type::kBOOL8;
};

enum class quant_type {
  kNONE,
  kASYMM,
  kSYMM,
  kSYMM_PCQ,
  kDFP,
  kINVALID,

};

static std::string as_string(quant_type q) {
  std::string str;
  switch (q) {
    case quant_type::kNONE:
      str = "none";
      break;
    case quant_type::kASYMM:
      str = "asymm";
      break;
    case quant_type::kSYMM:
      str = "symm";
      break;
    case quant_type::kSYMM_PCQ:
      str = "pcq_symm";
      break;
    case quant_type::kDFP:
      str = "dfp";
      break;
    default:
      str = "invalid";
      break;
  }
  return str;
};

enum class modifier {
  kOPTIONAL,
  kREQUIRED
};

enum class tensor_attr {
  kVARIABLE,
  kCONSTANT
};

struct tensor_tag {};
struct scalar_tag {};
struct none_type {};

template <typename T>
struct tag_of{
  using type = typename T::tag_type;
};

template <typename T>
using tag_of_t = typename tag_of<T>::type;

struct tensor_storage {
  // signature match required
  data_type dtype{data_type::kINVALID};
  quant_type qtype{quant_type::kNONE};
  tensor_attr attr{tensor_attr::kVARIABLE};

  // rule check required
  std::vector<uint32_t> shape;
  float scale{1.0f};
  int32_t zero_point{0};
  std::vector<float> per_channel_scales{};
  std::vector<int32_t> per_channel_zero_points{};
  uint32_t channel_dim;
  const void* data{nullptr};
  uint32_t data_length{0};
};

struct scalar_storage {
  // signature match required
  data_type dtype;

  // rule check required
  std::vector<uint8_t> data;
};

template <typename T>
struct trait_storage_type {
  using type = none_type;
};

template <>
struct trait_storage_type<tensor_tag> {
  using type = tensor_storage;
};

template <>
struct trait_storage_type<scalar_tag> {
  using type = scalar_storage;
};

template <typename T>
using is_tensor = std::is_same<T, tensor_tag>;

template <typename T>
using is_scalar = std::is_same<T, scalar_tag>;

template <typename T>
std::string role_name_as_string() {
    return std::string("not set up role name");
}

template <typename Field>
struct role_name {
    static const void* doc;
};

template <typename R /*role*/,
          typename T /*type*/,
          modifier M /*modifier*/,
          typename S = typename trait_storage_type<T>::type /*storage*/>
struct field {
  using Role = R;
  using type = T;
  static constexpr modifier m = M;
  field() {}

  void doc();

  std::string field_name;
  S storage;
  // type field_type;
};

template <typename R, modifier M>
struct tensor_field : public field<R, tensor_tag, M, tensor_storage> {
  tensor_field() {}

  tensor_field(const tensor_storage& storage) {
    this->storage.dtype = storage.dtype;
    this->storage.qtype = storage.qtype;
    this->storage.attr = storage.attr;
    this->storage.shape = storage.shape;
    this->storage.scale = storage.scale;
    this->storage.zero_point = storage.zero_point;
    this->storage.per_channel_scales = storage.per_channel_scales;
    this->storage.per_channel_zero_points = storage.per_channel_zero_points;
    this->storage.channel_dim = storage.channel_dim;
    this->storage.data = storage.data;
    this->storage.data_length = storage.data_length;
  }

  tensor_field(data_type d, quant_type q, tensor_attr a = tensor_attr::kVARIABLE) {
    this->storage.qtype = q;
    this->storage.dtype = d;
    this->storage.attr = a;
  }

  std::vector<uint32_t>& shape() { return this->storage.shape; }

  const void* data() { return this->storage.data; }

  uint32_t data_length() { return this->storage.data_length; }

  float& scale() { return this->storage.scale; }

  int& zero_ponit() { return this->storage.zero_point; }

  std::vector<float>& per_channel_scales() {
    return this->storage.per_channel_scales;
  }

  std::vector<int32_t>& per_channel_zero_points() {
    return this->storage.per_channel_zero_points;
  }

  uint32_t& channel_dim() { return this->storage.channel_dim; }

  void doc() {
    std::string name((const char*)R::doc);// = role_name_as_string<R>();
    std::cout << as_string(this->storage.dtype) << "/" << as_string(this->storage.qtype) << "|";
  }
};

template <typename R, modifier M>
struct scalar_field : public field<R, scalar_tag, M, scalar_storage> {
  scalar_field() {}

  scalar_field(const scalar_storage& storage) {
    this->storage.dtype = storage.dtype;
    this->storage.data = storage.data;
  }

  template <typename T>
  scalar_field(const T& v) {
    this->storage.dtype = value_type_convertor<T>::value;
    this->storage.data.resize(sizeof(v));
    memcpy(this->storage.data.data(), &v, sizeof(v));
  }

  template <typename T>
  scalar_field(const std::vector<T>& vec) {
    this->storage.dtype = value_type_convertor<T>::value;
    this->storage.data.resize(sizeof(T) * vec.size());
    memcpy(this->storage.data.data(), vec.data(), sizeof(T) * vec.size());
  };

  scalar_field(data_type d) {
    this->storage.dtype = d;
  }

  template <typename T>
  T get() {
    if (this->storage.dtype == value_type_convertor<T>::value) {
      std::cout << "get value from scalr is correct" << std::endl;
      return T(0);
    } else {
      std::cout << "get value from scalar_tag is type-mismatch" << std::endl;
      return T(0);
    }
  }

  void doc() {
    std::cout << as_string(this->storage.dtype) << "|";
    // << std::endl
    ;
  }
};


template <typename... F /*fileds*/>
struct signature {
  using type = std::tuple<F...>;

  signature() {}
  signature(const type& fields) : field_tuple(fields) {}

  template <typename Elem>
  inline Elem& get() {
    return std::get<Elem>(field_tuple);
  }

  template <typename Elem>
  inline const Elem& get() const {
    return std::get<Elem>(field_tuple);
  }

  void doc() {
    std::cout << "|" ;
    Tuple_For(field_tuple);
    std::cout << std::endl;
  }

  std::tuple<F...> field_tuple;
};

template <typename T>
struct rule {
  rule() : expr(T()) {}
  rule(T expr) : expr(expr) {}

  template <typename S /*signature type*/>
  typename T::rtype operator()(const S& sobj) {
    return expr(sobj);
  }

  void doc() {
    std::cout << "Document placeholder for rule" << std::endl;
  }

  T expr;
};
}  // namespace type
}  // namespace slang

#endif