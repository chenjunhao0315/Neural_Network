//
//  Mat.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/8/5.
//

#ifndef Mat_hpp
#define Mat_hpp

#include <iostream>
#include <stdio.h>
#include <cstring>
#include <initializer_list>
#include <iomanip>

#include "Cast.hpp"

#define MAT_FREE(data) if (data) delete data; data = nullptr;
#define MAT_FREE_ARRAY(data) if (data) delete [] data; data = nullptr;

using std::ostream;

enum MatType {
    MAT_8UC1 = 0,
    MAT_8UC3 = 1,
    MAT_8SC1 = 2,
    MAT_8SC3 = 3,
    MAT_32UC1 = 4,
    MAT_32UC3 = 5,
    MAT_32SC1 = 6,
    MAT_32SC3 = 7,
    MAT_32FC1 = 8,
    MAT_32FC3 = 9,
    MAT_UNDEFINED
};

class Mat;

template <typename T>
class Vec3;

struct Scalar {
    Scalar(float v0 = 0, float v1 = 0, float v2 = 0);
    float val[3];
};

void convertScalar(Scalar &s, const void *dst, MatType type);

typedef Vec3<unsigned char> Vec3b;
typedef Vec3<char> Vec3c;
typedef Vec3<int> Vec3i;
typedef Vec3<unsigned int> Vec3u;
typedef Vec3<float> Vec3f;

typedef void (*BinaryFunc)(void*, void*, int);
typedef void (*TernaryFunc)(void*, void*, void*, int);
typedef void (*MatrixFunc)(void*, void*, void*, int, int, int);

class Mat {
public:
    ~Mat();
    Mat(int width = 0, int height = 0, MatType type = MAT_8UC3, Scalar scalar = Scalar(0, 0, 0));
    Mat(int width, int height, MatType type, void *src);
    Mat(const Mat &M);
    Mat(Mat &&M);
    Mat& operator=(const Mat &M);
    Mat& operator=(std::initializer_list<float> list);
    
    friend ostream& operator<<(ostream& os, Mat& m);
    
    void release();
    int depth();
    int elemSize() {return step[1];}
    int getStep() {return step[0];}
    MatType getType() {return type;}
    unsigned char * ptr() {return data;}
    
    Mat convertTo(MatType dstType, float scale = 0, float shift = 0);
    
    Mat subtract(Mat &minuend, MatType dstType = MAT_UNDEFINED);
    Mat add(Mat &addend, MatType dstType = MAT_UNDEFINED);
    Mat divide(Mat &divisor, MatType dstType = MAT_UNDEFINED);
    Mat multiply(Mat &multiplier, MatType dstType = MAT_UNDEFINED);
    Mat scale(float scale, MatType dstType = MAT_UNDEFINED);
    Mat addWeighted(float alpha, Mat &addend, float beta, float gamma, MatType dstType);
    Mat absScale(float scale = 1, float alpha = 0);
    Mat transpose();
    Mat dot(Mat &m);
    
    void scale_channel(int channel, float scale);
    
    template <typename T>
    T* ptr(int index);
    
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
private:
    void copyFrom(void *src);
    unsigned char * allocData(int width, int height, MatType type);
    void copyData(unsigned char *dst, unsigned char *src, int width, int height, MatType type);
    void freeData();
    void stepCalculate();
    void fillWith(Scalar &s);
    
    int getIndex(int width, int height) const;
    
    int step[2];
    MatType type;
    unsigned char *data;
};

int getDepth(MatType type);

template <typename srcType, typename dstType>
void convertType(void *src, void *dst, int total_elements);

BinaryFunc getConvertTypeFunc(MatType srcType, MatType dstType);

template <typename srcType>
void absMat(void *src, void *dst, int total_elements);

BinaryFunc getabsMatFunc(MatType type);

template <typename srcType, typename dstType>
void subtractMat(void *src1, void *src2, void *dst, int total_elements);

TernaryFunc getsubtractMatFunc(MatType srcType, MatType dstType);

template <typename srcType, typename dstType>
void addMat(void *src1, void *src2, void *dst, int total_elements);

TernaryFunc getaddMatFunc(MatType srcType, MatType dstType);

template <typename srcType, typename dstType>
void divideMat(void *src1, void *src2, void *dst, int total_elements);

TernaryFunc getdivideMatFunc(MatType srcType, MatType dstType);

template <typename srcType, typename dstType>
void multiplyMat(void *src1, void *src2, void *dst, int total_elements);

TernaryFunc getmultiplyMatFunc(MatType srcType, MatType dstType);

template <typename srcType, typename dstType>
void dotMat(void *src1, void *src2, void *dst, int m, int n, int k);

MatrixFunc getDotFunc(MatType srcType, MatType dstType);


template <typename T>
class Vec3 {
public:
    Vec3(T data_0 = 0, T data_1 = 0, T data_2 = 0);
    T& operator[](int index);
    const T& operator[](int index) const;
    Vec3& operator-=(const Vec3 &a);
    Vec3& operator+=(const Vec3 &a);
    Vec3& operator=(const Vec3 &a);
private:
    T data[3];
};

class ConvTool {
public:
    ConvTool(MatType srcType_, MatType dstType_, Mat &kernel_) : srcType(srcType_), dstType(dstType_), kernel(kernel_) {}
    void start(Mat &src, Mat &dst, int channel = -1);
private:
    MatType srcType;
    MatType dstType;
    Mat kernel;
};

template <typename srcType, typename dstType>
class ConvEngine {
public:
    ConvEngine(int dst_channel_, int dst_width_, int dst_height_, int src_channel_,  int src_width_, int kernel_size_) : dst_channel(dst_channel_), dst_width(dst_width_), dst_height(dst_height_), src_channel(src_channel_), src_width(src_width_), kernel_size(kernel_size_) {}
    void process(void *src, void *dst, void *kernel, int channel = -1);
private:
    void convSize1(void *src, void *dst, void *kernel);
    void convSize1_c(void *src, void *dst, void *kernel, int channel);
    void convDefinition(void *src, void *dst, void *kernel);
    int dst_channel;
    int dst_width;
    int dst_height;
    int src_channel;
    int src_width;
    int kernel_size;
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
T* Mat::ptr(int index) {
    return (T*)(data + step[0] * index);
}

template <typename T>
typename Mat::MatIterator<T> Mat::begin() {
    return typename Mat::MatIterator<T>((T*)data);
}

template <typename T>
typename Mat::MatIterator<T> Mat::end() {
    return typename Mat::MatIterator<T>((T*)(data + step[0] * height));
}

template <typename srcType, typename dstType>
void convertType(void *src, void *dst, int total_elements) {
    srcType *src_ptr = (srcType*)src;
    dstType *dst_ptr = (dstType*)dst;
    for (int i = total_elements; i--; ) {
        *(dst_ptr++) = saturate_cast<dstType>(*(src_ptr++));
    }
}

template <typename srcType>
void absMat(void *src, void *dst, int total_elements) {
    srcType *src_ptr = (srcType*)src;
    srcType *dst_ptr = (srcType*)dst;
    for (int i = total_elements; i--; ) {
        *(dst_ptr++) = saturate_cast<srcType>(abs(*(src_ptr++)));
    }
}

template <typename srcType, typename dstType>
void subtractMat(void *src1, void *src2, void *dst, int total_elements) {
    srcType *src1_ptr = (srcType*)src1;
    dstType *src2_ptr = (dstType*)src2;
    dstType *dst_ptr = (dstType*)dst;
    for (int i = total_elements; i--; ) {
        *(dst_ptr++) = saturate_cast<dstType>(*(src1_ptr++) - *(src2_ptr++));
    }
}

template <typename srcType, typename dstType>
void addMat(void *src1, void *src2, void *dst, int total_elements) {
    srcType *src1_ptr = (srcType*)src1;
    dstType *src2_ptr = (dstType*)src2;
    dstType *dst_ptr = (dstType*)dst;
    for (int i = total_elements; i--; ) {
        *(dst_ptr++) = saturate_cast<dstType>(*(src1_ptr++) + *(src2_ptr++));
    }
}

template <typename srcType, typename dstType>
void divideMat(void *src1, void *src2, void *dst, int total_elements) {
    srcType *src1_ptr = (srcType*)src1;
    dstType *src2_ptr = (dstType*)src2;
    dstType *dst_ptr = (dstType*)dst;
    for (int i = total_elements; i--; ) {
        *(dst_ptr++) = saturate_cast<dstType>(*(src1_ptr++) / *(src2_ptr++));
    }
}

template <typename srcType, typename dstType>
void multiplyMat(void *src1, void *src2, void *dst, int total_elements) {
    srcType *src1_ptr = (srcType*)src1;
    dstType *src2_ptr = (dstType*)src2;
    dstType *dst_ptr = (dstType*)dst;
    for (int i = total_elements; i--; ) {
        *(dst_ptr++) = saturate_cast<dstType>(*(src1_ptr++) * *(src2_ptr++));
    }
}

// src2 needs to be transposed first
template <typename srcType, typename dstType>
void dotMat(void *src1, void *src2, void *dst, int m, int n, int k) {
//    srcType *src1_ptr = (srcType*)src1;
//    dstType *src2_ptr = (dstType*)src2;
//    dstType *dst_ptr = (dstType*)dst;
    
    
}

// End Mat

// ConvEngine
template <typename srcType, typename dstType>
void ConvEngine<srcType, dstType>::process(void *src, void *dst, void *kernel, int channel) {
    if (kernel_size == 1) {
        if (channel == -1)
            convSize1(src, dst, kernel);
        else
            convSize1_c(src, dst, kernel, channel);
    } else {
        convDefinition(src, dst, kernel);
    }
}

template <typename srcType, typename dstType>
void ConvEngine<srcType, dstType>::convDefinition(void *src, void *dst, void *kernel_) {
    int padding = (kernel_size - 1) / 2;
    int x, y;
    int coordinate_w, coordinate_h;
    srcType *src_ptr = (srcType*)src;
    dstType *dst_ptr = (dstType*)dst;
    float conv[3] = {0};
    int src_index;
    float *kernel = (float*)kernel_;
    float *kernel_ptr;
    
    y = -padding;
    for (int h = 0; h < dst_height; ++h, ++y) {
        x = -padding;
        for (int w = 0; w < dst_width; ++w, ++x) {
            kernel_ptr = kernel;
            conv[0] = 0; conv[1] = 0; conv[2] = 0;
            for (int kernel_h = 0; kernel_h < kernel_size; ++kernel_h) {
                coordinate_h = y + kernel_h;
                for (int kernel_w = 0; kernel_w < kernel_size; ++kernel_w) {
                    coordinate_w = x + kernel_w;
                    if (coordinate_w >= 0 && coordinate_w < dst_width && coordinate_h >= 0 && coordinate_h < dst_height) {
                        src_index = (coordinate_h * src_width + coordinate_w) * src_channel;
                        for (int c = 0; c < src_channel; ++c) {
                            conv[c] += *(src_ptr + src_index + c) * *(kernel_ptr);
                        }
                    }
                    kernel_ptr++;
                }
            }
            for (int c = 0; c < dst_channel; ++c) {
                *(dst_ptr++) = saturate_cast<dstType>(conv[c]);
            }
        }
    }
}

template <typename srcType, typename dstType>
void ConvEngine<srcType, dstType>::convSize1(void *src, void *dst, void *kernel) {
    srcType *src_ptr = (srcType*)src;
    dstType *dst_ptr = (dstType*)dst;
    float kernel_value = *(float *)kernel;
    for (int i = dst_width * dst_height * dst_channel; i--; ) {
        *(dst_ptr++) = saturate_cast<dstType>(*(src_ptr++) * kernel_value);
    }
}

template <typename srcType, typename dstType>
void ConvEngine<srcType, dstType>::convSize1_c(void *src, void *dst, void *kernel, int channel) {
    srcType *src_ptr = (srcType*)src;
    dstType *dst_ptr = (dstType*)dst;
    float kernel_value = *(float *)kernel;
    int counter = dst_channel - channel;
    for (int i = dst_width * dst_height * dst_channel; i--; ) {
        if ((counter++) % dst_channel == 0)
            *(dst_ptr++) = saturate_cast<dstType>(*(src_ptr++) * kernel_value);
        else
            *(dst_ptr++) = saturate_cast<dstType>(*(src_ptr++));
    }
}

// End ConvEngine

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
Vec3<T>& Vec3<T>::operator=(const Vec3<T> &a) {
    data[0] = a[0];
    data[1] = a[1];
    data[2] = a[2];
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
