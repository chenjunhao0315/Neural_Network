//
//  OtterMaybeOwned.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/12/17.
//

#ifndef OtterMaybeOwned_hpp
#define OtterMaybeOwned_hpp

#include <stdio.h>
#include <cstddef>
#include <type_traits>

namespace otter {

struct in_place_t {
    explicit in_place_t() = default;
};

template <std::size_t I>
struct in_place_index_t {
    explicit in_place_index_t() = default;
};

template <typename T>
struct in_place_type_t {
    explicit in_place_type_t() = default;
};

constexpr in_place_t in_place{};

template <typename T>
struct OtterMaybeOwnedTraitsNucleus {
    using owned_type = T;
    using borrow_type = const T *;
    
    static borrow_type create_borrow(const owned_type& from) {
        return &from;
    }
    
    static void assign_borrow(borrow_type& lhs, borrow_type rhs) {
        lhs = rhs;
    }
    
    static void destory_borrow(borrow_type& target) {}
    
    static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
        return *borrow;
    }
    
    static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
        return borrow;
    }
};

template <typename T>
struct OtterMaybeOwnedTraits;


template <typename T>
class OtterMaybeOwned {
private:
    using owned_type = typename OtterMaybeOwnedTraits<T>::owned_type;
    using borrow_type = typename OtterMaybeOwnedTraits<T>::borrow_type;
    
    bool isBorrowed_;
    union {
        owned_type own_;
        borrow_type borrow_;
    };
    
    explicit OtterMaybeOwned(const owned_type& t)
    : isBorrowed_(true), borrow_(OtterMaybeOwnedTraits<T>::create_borrow(t)) {}
    
    explicit OtterMaybeOwned(T&& t) noexcept(std::is_nothrow_move_constructible<T>::value) : isBorrowed_(false), own_(std::move(t)) {}
    
    template <class... Args>
    explicit OtterMaybeOwned(in_place_t, Args&&... args) : isBorrowed_(false), own_(std::forward<Args>(args)...) {}
    
public:
    OtterMaybeOwned() : isBorrowed_(true), borrow_() {}
    
    OtterMaybeOwned(const OtterMaybeOwned& rhs) : isBorrowed_(rhs.isBorrowed_) {
        if (rhs.isBorrowed_) {
            OtterMaybeOwnedTraits<T>::assign_borrow(borrow_, rhs.borrow_);
        } else {
            ::new (&own_) T(rhs.own_);
        }
    }
    
    OtterMaybeOwned& operator=(const OtterMaybeOwned& rhs) {
        if (this == &rhs) {
            return *this;
        }
        if (!isBorrowed_) {
            if (rhs.isBorrowed_) {
                own_.~T();
                OtterMaybeOwnedTraits<T>::assign_borrow(borrow_, rhs.borrow_);
                isBorrowed_ = true;
            } else {
                own_ = rhs.own_;
            }
        } else {
            if (rhs.isBorrowed_) {
                OtterMaybeOwnedTraits<T>::assign_borrow(borrow_, rhs.borrow_);
            } else {
                OtterMaybeOwnedTraits<T>::destroy_borrow(borrow_);
                new (&own_) T(rhs.own_);
                isBorrowed_ = false;
            }
        }
        //        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isBorrowed_ == rhs.isBorrowed_);
        return *this;
    }
    
    OtterMaybeOwned(OtterMaybeOwned&& rhs) noexcept(std::is_nothrow_move_constructible<T>::value) : isBorrowed_(rhs.isBorrowed_) {
        if (rhs.isBorrowed_) {
            OtterMaybeOwnedTraits<T>::assign_borrow(borrow_, rhs.borrow_);
        } else {
            new (&own_) T(std::move(rhs.own_));
        }
    }
    
    OtterMaybeOwned& operator=(OtterMaybeOwned&& rhs) noexcept(std::is_nothrow_move_assignable<T>::value) {
        if (this == &rhs) {
            return *this;
        }
        if (!isBorrowed_) {
            if (rhs.isBorrowed_) {
                own_.~T();
                OtterMaybeOwnedTraits<T>::assign_borrow(borrow_, rhs.borrow_);
                isBorrowed_ = true;
            } else {
                own_ = std::move(rhs.own_);
            }
        } else {
            if (rhs.isBorrowed_) {
                OtterMaybeOwnedTraits<T>::assign_borrow(borrow_, rhs.borrow_);
            } else {
                OtterMaybeOwnedTraits<T>::destroy_borrow(borrow_);
                new (&own_) T(std::move(rhs.own_));
                isBorrowed_ = false;
            }
        }
        //        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isBorrowed_ == rhs.isBorrowed_);
        return *this;
    }
    
    static OtterMaybeOwned borrowed(const T& t) {
        return OtterMaybeOwned(t);
    }
    
    static OtterMaybeOwned owned(T&& t) noexcept(std::is_nothrow_move_constructible<T>::value) {
        return OtterMaybeOwned(std::move(t));
    }
    
    template <class... Args>
    static OtterMaybeOwned owned(in_place_t, Args&&... args) {
        return OtterMaybeOwned(in_place, std::forward<Args>(args)...);
    }
    
    ~OtterMaybeOwned() {
        if (!isBorrowed_) {
            own_.~T();
        } else {
            OtterMaybeOwnedTraits<T>::destroy_borrow(borrow_);
        }
    }
    
    bool unsafeIsBorrowed() const {
        return isBorrowed_;
    }
    
    const T& operator*() const& {
        return (isBorrowed_) ? OtterMaybeOwnedTraits<T>::referenceFromBorrow(borrow_) : own_;
    }
        
    const T* operator->() const {
        return (isBorrowed_) ? OtterMaybeOwnedTraits<T>::pointerFromBorrow(borrow_) : &own_;
    }
        
    T operator*() && {
        if (isBorrowed_) {
            return OtterMaybeOwnedTraits<T>::referenceFromBorrow(borrow_);
        } else {
            return std::move(own_);
        }
    }
};
    
}   // end namespace otter

#endif /* OtterMaybeOwned_hpp */
