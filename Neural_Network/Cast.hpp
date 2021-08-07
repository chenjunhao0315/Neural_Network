//
//  Cast.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/8/7.
//

#ifndef Cast_h
#define Cast_h

// Cast
template<typename _Tp>
static inline _Tp saturate_cast(unsigned char v) {
    return _Tp(v);
}
template<typename _Tp>
static inline _Tp saturate_cast(char v) {
    return _Tp(v);
}
template<typename _Tp>
static inline _Tp saturate_cast(int v) {
    return _Tp(v);
}

template<typename _Tp>
static inline _Tp saturate_cast(unsigned int v) {
    return _Tp(v);
}

template<typename _Tp>
static inline _Tp saturate_cast(float v) {
    return _Tp(v);
}
template<typename _Tp>
static inline _Tp saturate_cast(double v) {
    return _Tp(v);
}
template<>
inline unsigned char saturate_cast<unsigned char>(char v) {
    return (unsigned char)std::max((int)v, 0);
}
template<>
inline unsigned char saturate_cast<unsigned char>(int v) {
    return (unsigned char)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0);
}
template<>
inline unsigned char saturate_cast<unsigned char>(float v) {
    int iv = round(v); return saturate_cast<unsigned char>(iv);
}
template<>
inline unsigned char saturate_cast<unsigned char>(double v) {
    int iv = round(v); return saturate_cast<unsigned char>(iv);
}
template<>
inline char saturate_cast<char>(unsigned char v) {
    return (char)std::min((int)v, SCHAR_MAX);
}
template<>
inline char saturate_cast<char>(int v) {
    return (char)((unsigned)(v-SCHAR_MIN) <= (unsigned)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN);
}
template<>
inline char saturate_cast<char>(float v) {
    int iv = round(v);
    return saturate_cast<char>(iv);
}
template<>
inline int saturate_cast<int>(float v) {
    return round(v);
}
template<>
inline int saturate_cast<int>(double v) {
    return round(v);
}

template<>
inline unsigned saturate_cast<unsigned>(float v) {
    return round(v);
}

template<>
inline unsigned saturate_cast<unsigned>(double v) {
    return round(v);
}
// End Cast

#endif /* Cast_h */
