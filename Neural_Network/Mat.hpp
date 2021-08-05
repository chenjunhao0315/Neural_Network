//
//  Mat.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/8/5.
//

#ifndef Mat_hpp
#define Mat_hpp

#include <stdio.h>
#include <cstring>

class Mat;

template <typename T>
class Vec3;

typedef Vec3<unsigned char> Vec3b;
typedef Vec3<char> Vec3c;
typedef Vec3<int> Vec3i;
typedef Vec3<unsigned int> Vec3u;
typedef Vec3<float> Vec3f;

template <typename T>
class Vec3 {
public:
    Vec3(T data_0 = 0, T data_1 = 0, T data_2 = 0);
    T& operator[](int index);
    const T& operator[](int index) const;
    Vec3& operator-=(const Vec3 &a);
    Vec3& operator+=(const Vec3 &a);
private:
    T data[3];
};

class Mat {
public:
    enum MatType {
        MAT_8UC1,
        MAT_8UC3,
        MAT_8SC1,
        MAT_8SC3,
        MAT_32UC1,
        MAT_32UC3,
        MAT_32SC1,
        MAT_32SC3,
        MAT_32FC1,
        MAT_32FC3
    };
    ~Mat();
    Mat(int width = 0, int height = 0, MatType type = MAT_8UC3);
    Mat(const Mat &M);
    Mat(Mat &&M);
    Mat& operator=(const Mat &M);
    void stepCalculate();
    unsigned char * allocData(int width, int height, MatType type);
    void copyData(unsigned char *dst, unsigned char *src, int width, int height, MatType type);
    void freeData();
    int getIndex(int width, int height) const;
    void show();
    
    unsigned char * ptr() {return data;}
    int elemSize() {return step[1];}
    
    template<typename T>
    T& at(int index);
    
    template<typename T>
    T& at(int width_, int height_);
    
    template <typename T>
    class MatIterator{
    public:
        MatIterator(T *start) : current(start) {}
        T& operator*() const { return *(T*)current;}
        T* operator->() const { return (T*)current;}
        MatIterator& operator++();
        MatIterator operator++(int);
        bool operator!=(const MatIterator right) const;
        bool operator==(const MatIterator right) const;
        
    private:
        T *current;
    };
    
    template <typename T>
    MatIterator<T> begin();
    
    template <typename T>
    MatIterator<T> end();
    
    int width;
    int height;
    int step[2];
    MatType type;
    unsigned char *data;
};

// Mat
template<typename T>
T& Mat::at(int index) {
    return *(T*)(data + index * step[0]);
}

template<typename T>
T& Mat::at(int width_, int height_) {
    return *(T*)(data + getIndex(width_, height_));
}

template <typename T>
typename Mat::MatIterator<T> Mat::begin() {
    return typename Mat::MatIterator<T>((T*)data);
}

template <typename T>
typename Mat::MatIterator<T> Mat::end() {
    return typename Mat::MatIterator<T>((T*)(data + step[0] * height));
}
// End Mat

// Mat Iterator
template <typename T>
typename Mat::MatIterator<T>& Mat::MatIterator<T>::operator++() {
    current++; return *this;
}

template <typename T>
typename Mat::MatIterator<T> Mat::MatIterator<T>::operator++(int) {
    MatIterator old = *this;
    current++;
    return old;
}

template <typename T>
bool Mat::MatIterator<T>::operator!=(const MatIterator<T> right) const {
    return current != right.current;
}

template <typename T>
bool Mat::MatIterator<T>::operator==(const MatIterator<T> right) const {
    return current == right.current;
}

// End Mat Iterator

// Vec3
template <typename T>
Vec3<T>::Vec3(T data_0, T data_1, T data_2) {
    data[0] = data_0;
    data[1] = data_1;
    data[2] = data_2;
}

template <typename T>
T& Vec3<T>::operator[](int index) {
    return data[index];
}

template <typename T>
const T& Vec3<T>::operator[](int index) const {
    return data[index];
}

template <typename T>
Vec3<T>& Vec3<T>::operator-=(const Vec3<T> &a) {
    data[0] -= a[0];
    data[1] -= a[1];
    data[2] -= a[2];
    return *this;
}

template <typename T>
Vec3<T>& Vec3<T>::operator+=(const Vec3<T> &a) {
    data[0] += a[0];
    data[1] += a[1];
    data[2] += a[2];
    return *this;
}

template <typename T>
Vec3<T> operator+(const Vec3<T> &a, const Vec3<T> &b) {
    return Vec3<T>(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

template <typename T>
Vec3<T> operator-(const Vec3<T> &a, const Vec3<T> &b) {
    return Vec3<T>(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
// End Vec3

#endif /* Mat_hpp */
