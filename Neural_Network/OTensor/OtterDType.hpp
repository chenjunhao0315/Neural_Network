//
//  OtterDType.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/12/4.
//

#ifndef OtterDType_hpp
#define OtterDType_hpp

#include <stdio.h>
#include <string>

class _Uninitialized {};

#define DTYPE_CONVERSION_TABLE(_)       \
    _(uint8_t, Byte)      /* 0 */       \
    _(int8_t, Char)       /* 1 */       \
    _(int16_t, Short)     /* 2 */       \
    _(int, Int)           /* 3 */       \
    _(int64_t, Long)      /* 4 */       \
    _(float, Float)       /* 5 */       \
    _(double, Double)     /* 6 */       \
    _(bool, Bool)         /* 7 */

enum class ScalarType : int8_t {
#define DEFINE_ENUM(_1, n) n,
    DTYPE_CONVERSION_TABLE(DEFINE_ENUM)
#undef DEFINE_ENUM
    Undefined,
    NumOptions
};

std::string toString(ScalarType type);

struct OtterTypeData final {
    using New = void*();
    using PlacementNew = void(void*, size_t);
    using Copy = void(const void*, void*, size_t);
    using PlacementDelete = void(void*, size_t);
    using Delete = void(void*);
    
    constexpr OtterTypeData() noexcept
          : itemsize_(0),
            new_(nullptr),
            placementNew_(nullptr),
            copy_(nullptr),
            placementDelete_(nullptr),
            delete_(nullptr) {}
    
    constexpr OtterTypeData(
          size_t itemsize,
          New* newFn,
          PlacementNew* placementNew,
          Copy* copy,
          PlacementDelete* placementDelete,
          Delete* deleteFn) noexcept
          : itemsize_(itemsize),
            new_(newFn),
            placementNew_(placementNew),
            copy_(copy),
            placementDelete_(placementDelete),
            delete_(deleteFn) {}
    
    size_t itemsize_;
    New* new_;
    PlacementNew* placementNew_;
    Copy* copy_;
    PlacementDelete* placementDelete_;
    Delete* delete_;
};

template <typename T>
inline void* _New() {
    return new T;
}

template <typename T, std::enable_if_t<std::is_default_constructible<T>::value>* = nullptr>
inline constexpr OtterTypeData::New* _PickNew() {
    return &_New<T>;
}

template <typename T>
inline void _PlacementNew(void* ptr, size_t n) {
    T* typed_ptr = static_cast<T*>(ptr);
    for (size_t i = 0; i < n; ++i) {
        new (typed_ptr + i) T;
    }
}

template <typename T, std::enable_if_t<std::is_default_constructible<T>::value>* = nullptr>
inline constexpr OtterTypeData::PlacementNew* _PickPlacementNew() {
    return (std::is_fundamental<T>::value || std::is_pointer<T>::value)
        ? nullptr
        : &_PlacementNew<T>;
}

template <typename T>
inline void _Copy(const void* src, void* dst, size_t n) {
    const T* typed_src = static_cast<const T*>(src);
    T* typed_dst = static_cast<T*>(dst);
    for (size_t i = 0; i < n; ++i) {
        typed_dst[i] = typed_src[i];
    }
}

template <typename T, std::enable_if_t<std::is_copy_assignable<T>::value>* = nullptr>
inline constexpr OtterTypeData::Copy* _PickCopy() {
    return (std::is_fundamental<T>::value || std::is_pointer<T>::value)
        ? nullptr
        : &_Copy<T>;
}

template <typename T>
inline void _PlacementDelete(void* ptr, size_t n) {
    T* typed_ptr = static_cast<T*>(ptr);
    for (size_t i = 0; i < n; ++i) {
        typed_ptr[i].~T();
    }
}

template <typename T>
inline constexpr OtterTypeData::PlacementDelete* _PickPlacementDelete() {
    return (std::is_fundamental<T>::value || std::is_pointer<T>::value)
        ? nullptr
        : &_PlacementDelete<T>;
}

template <typename T>
inline void _Delete(void* ptr) {
    T* typed_ptr = static_cast<T*>(ptr);
    delete typed_ptr;
}

template <class T>
inline constexpr OtterTypeData::Delete* _PickDelete() noexcept {
    return &_Delete<T>;
}

struct OtterType {
public:
    using New = OtterTypeData::New;
    using PlacementNew = OtterTypeData::PlacementNew;
    using Copy = OtterTypeData::Copy;
    using PlacementDelete = OtterTypeData::PlacementDelete;
    using Delete = OtterTypeData::Delete;
    
//    OtterType() {}
    
    OtterType() noexcept;
    OtterType(const OtterType& src) noexcept = default;
    OtterType& operator=(const OtterType& src) noexcept = default;
    OtterType(OtterType&& rhs) noexcept = default;
    inline OtterType& operator=(ScalarType scalar_type) noexcept {
        index_ = static_cast<uint16_t>(scalar_type);
        return *this;
    }
    
    friend bool operator==(const OtterType lhs, const OtterType rhs) noexcept;
    
    #define MaxTypeIndex 32
    static OtterTypeData* typeMetaDatas();
    
    template <typename T>
    static OtterType Make() {
        return OtterType(type2index<T>());
    }
    
    template <typename T>
    bool Match() const noexcept {
        return (*this == Make<T>());
    }
    
    inline size_t itemsize() const noexcept {
        return data().itemsize_;
    }
    
    New* newFn() const noexcept {
        return data().new_;
    }
    
    PlacementNew* placementNew() const noexcept {
        return data().placementNew_;
    }
    
    Copy* copy() const noexcept {
        return data().copy_;
    }
    
    PlacementDelete* placementDelete() const noexcept {
        return data().placementDelete_;
    }
    
    Delete* deleteFn() const noexcept {
        return data().delete_;
    }
    
    template <class T>
    static constexpr size_t ItemSize() noexcept {
        return sizeof(T);
    }
    
    static inline OtterType fromScalarType(ScalarType scalar_type) {
        const auto index = static_cast<uint16_t>(scalar_type);
        return OtterType(index);
    }

    inline ScalarType toScalarType() {
        return static_cast<ScalarType>(index_);
    }
private:
    explicit OtterType(const uint16_t index) noexcept : index_(index) {}
    
    template <class T>
    static uint16_t type2index() noexcept;
    
    inline const OtterTypeData& data() const {
        return typeMetaDatas()[index_];
    }
    
    uint16_t index_;
};

#define DEFINE_SCALAR_METADATA_INSTANCE(T, name)                \
    template <>                                                 \
    constexpr uint16_t OtterType::type2index<T>() noexcept {     \
        return static_cast<uint16_t>(ScalarType::name);         \
    }
    DTYPE_CONVERSION_TABLE(DEFINE_SCALAR_METADATA_INSTANCE)
#undef DEFINE_SCALAR_METADATA_INSTANCE

template <>
constexpr uint16_t OtterType::type2index<_Uninitialized>() noexcept {
    return static_cast<uint16_t>(ScalarType::Undefined);
}

inline OtterType::OtterType() noexcept : index_(type2index<_Uninitialized>()) {}

inline bool operator==(const OtterType lhs, const OtterType rhs) noexcept {
    return (lhs.index_ == rhs.index_);
}

inline bool operator!=(const OtterType lhs, const OtterType rhs) noexcept {
    return !operator==(lhs, rhs);
}

#endif /* OtterDType_hpp */
