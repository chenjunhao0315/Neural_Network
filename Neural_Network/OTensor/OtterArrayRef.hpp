//
//  OtterArrayRef.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/12/7.
//

#ifndef OtterArrayRef_hpp
#define OtterArrayRef_hpp

#include <stdio.h>
#include <vector>
#include <initializer_list>
#include <iostream>

template <typename T>
class OtterArrayRef {
public:
    using iterator = const T*;
    using const_iterator = const T*;
    using value_type = T;
    
    constexpr OtterArrayRef() : data_(nullptr), length_(0) {}
    constexpr OtterArrayRef(T& single) : data_(&single), length_(1) {}
    OtterArrayRef(const T* data, size_t length) : data_(data), length_(length) {}
    OtterArrayRef(const T* begin, const T* end) : data_(begin), length_(end - begin) {}
    
    template <typename A>
    OtterArrayRef(const std::vector<T, A>& vec) : data_(vec.data()), length_(vec.size()) {}
    
    constexpr OtterArrayRef(const std::initializer_list<T>& vec) : data_((std::begin(vec) == std::end(vec)) ? static_cast<T*>(nullptr) : std::begin(vec)), length_(vec.size()) {}
    
    template <size_t N>
    constexpr OtterArrayRef(const T (&Arr)[N]) : data_(Arr), length_(N) {}
    
    constexpr iterator begin() const {
        return data_;
    }
    
    constexpr iterator end() const {
        return data_ + length_;
    }
    
    constexpr const_iterator const_begin() const {
        return data_;
    }
    
    constexpr const_iterator const_end() const {
        return data_ + length_;
    }
    
    constexpr bool empty() const {
        return length_ == 0;
    }
    
    constexpr const T* data() const {
        return data_;
    }
    
    constexpr size_t size() const {
        return length_;
    }
    
    const T& front() const {
        if (empty()) fprintf(stderr, "[ArrayRef] Empty array!\n");
        return data_[0];
    }
    
    const T& back() const {
        if (empty()) fprintf(stderr, "[ArrayRef] Empty array!\n");
        return data_[length_ - 1];;
    }
    
    OtterArrayRef<T> slice(size_t N, size_t M) const {
        return OtterArrayRef<T>(data() + N, M);
    }
    
    constexpr const T& operator[](size_t idx) const {
        return data_[idx];
    }
    
    std::vector<T> vec() {
        return std::vector<T>(data_, data_ + length_);
    }
private:
    const T* data_;
    size_t length_;
};

template <typename T>
std::ostream& operator<<(std::ostream& out, OtterArrayRef<T> list) {
    int i = 0;
    out << "[";
    for (auto e : list) {
        if (i++ > 0)
            out << ", ";
        out << e;
    }
    out << "]";
    return out;
}

using IntArrayRef = OtterArrayRef<int64_t>;

#endif /* OtterArrayRef_hpp */
