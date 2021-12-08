//
//  OtterAccumulator.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/12/9.
//

#ifndef OtterAccumulator_hpp
#define OtterAccumulator_hpp

#include <stdio.h>
#include <iterator>
#include <numeric>
#include <type_traits>

#endif /* OtterAccumulator_hpp */

template <typename C, typename std::enable_if<std::is_integral<typename C::value_type>::value, int>::type = 0>
inline int64_t sum_integers(const C& container) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(container.begin(), container.end(), static_cast<int64_t>(0));
}

/// Sum of integer elements referred to by iterators; accumulates into the
/// int64_t datatype
template <
    typename Iter,
    typename std::enable_if<
        std::is_integral<
            typename std::iterator_traits<Iter>::value_type>::value,
        int>::type = 0>
inline int64_t sum_integers(Iter begin, Iter end) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(begin, end, static_cast<int64_t>(0));
}

/// Product of a list of integers; accumulates into the int64_t datatype
template <
    typename C,
    typename std::enable_if<
        std::is_integral<typename C::value_type>::value,
        int>::type = 0>
inline int64_t multiply_integers(const C& container) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(
      container.begin(),
      container.end(),
      static_cast<int64_t>(1),
      std::multiplies<int64_t>());
}

/// Product of integer elements referred to by iterators; accumulates into the
/// int64_t datatype
template <
    typename Iter,
    typename std::enable_if<
        std::is_integral<
            typename std::iterator_traits<Iter>::value_type>::value,
        int>::type = 0>
inline int64_t multiply_integers(Iter begin, Iter end) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(
      begin, end, static_cast<int64_t>(1), std::multiplies<int64_t>());
}

/// Return product of all dimensions starting from k
/// Returns 1 if k>=dims.size()
template <
    typename C,
    typename std::enable_if<
        std::is_integral<typename C::value_type>::value,
        int>::type = 0>
inline int64_t numelements_from_dim(const int k, const C& dims) {
  if (k > dims.size()) {
    return 1;
  } else {
    auto cbegin = dims.cbegin();
    std::advance(cbegin, k);
    return multiply_integers(cbegin, dims.cend());
  }
}

/// Product of all dims up to k (not including dims[k])
/// Throws an error if k>dims.size()
template <
    typename C,
    typename std::enable_if<
        std::is_integral<typename C::value_type>::value,
        int>::type = 0>
inline int64_t numelements_to_dim(const int k, const C& dims) {
  auto cend = dims.cbegin();
  std::advance(cend, k);
  return multiply_integers(dims.cbegin(), cend);
}


template <typename C, typename std::enable_if<std::is_integral<typename C::value_type>::value, int>::type = 0>
inline int64_t numelements_between_dim(int k, int l, const C& dims) {
  if (k > l) {
    std::swap(k, l);
  }

  TORCH_INTERNAL_ASSERT((unsigned)l < dims.size());

  auto cbegin = dims.cbegin();
  auto cend = dims.cbegin();
  std::advance(cbegin, k);
  std::advance(cend, l);
  return multiply_integers(cbegin, cend);
}
