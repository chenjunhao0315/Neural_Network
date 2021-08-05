//
//  Mat.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/8/5.
//

#include "Mat.hpp"

Mat::~Mat() {
    freeData();
}

void Mat::freeData() {
    delete [] data;
}

unsigned char * Mat::allocData(int width, int height, MatType type) {
    switch (type) {
        case MAT_8UC1:
            return new unsigned char [width * height]; break;
        case MAT_8UC3:
            return (unsigned char *)(new Vec3b [width * height]); break;
        case MAT_8SC1:
            return (unsigned char *)(new char [width * height]); break;
        case MAT_8SC3:
            return (unsigned char *)(new Vec3c [width * height]); break;
        case MAT_32UC1:
            return (unsigned char *)(new unsigned int [width * height]); break;
        case MAT_32UC3:
            return (unsigned char *)(new Vec3u [width * height]); break;
        case MAT_32SC1:
            return (unsigned char *)(new int [width * height]); break;
        case MAT_32SC3:
            return (unsigned char *)(new Vec3i [width * height]); break;
        case MAT_32FC1:
            return (unsigned char *)(new float [width * height]); break;
        case MAT_32FC3:
            return (unsigned char *)(new Vec3f [width * height]); break;
    }
    return nullptr;
}

void Mat::copyData(unsigned char *dst, unsigned char *src, int width, int height, MatType type) {
    switch (type) {
        case MAT_8UC1:
            memcpy(dst, src, sizeof(unsigned char) * width * height); break;
        case MAT_8UC3:
            memcpy(dst, src, sizeof(Vec3b) * width * height); break;
        case MAT_8SC1:
            memcpy(dst, src, sizeof(char) * width * height); break;
        case MAT_8SC3:
            memcpy(dst, src, sizeof(Vec3c) * width * height); break;
        case MAT_32UC1:
            memcpy(dst, src, sizeof(unsigned int) * width * height); break;
        case MAT_32UC3:
            memcpy(dst, src, sizeof(Vec3u) * width * height); break;
        case MAT_32SC1:
            memcpy(dst, src, sizeof(int) * width * height); break;
        case MAT_32SC3:
            memcpy(dst, src, sizeof(Vec3i) * width * height); break;
        case MAT_32FC1:
            memcpy(dst, src, sizeof(float) * width * height); break;
        case MAT_32FC3:
            memcpy(dst, src, sizeof(Vec3f) * width * height); break;
    }
}

void Mat::stepCalculate() {
    switch (type) {
        case MAT_8UC1:
            step[1] = sizeof(unsigned char); break;
        case MAT_8UC3:
            step[1] = sizeof(Vec3b); break;
        case MAT_8SC1:
            step[1] = sizeof(char); break;
        case MAT_8SC3:
            step[1] = sizeof(Vec3c); break;
        case MAT_32UC1:
            step[1] = sizeof(unsigned int); break;
        case MAT_32UC3:
            step[1] = sizeof(Vec3u); break;
        case MAT_32SC1:
            step[1] = sizeof(int); break;
        case MAT_32SC3:
            step[1] = sizeof(Vec3i); break;
        case MAT_32FC1:
            step[1] = sizeof(float); break;
        case MAT_32FC3:
            step[1] = sizeof(Vec3f); break;
    }
    step[0] = width * step[1];
}

Mat::Mat(int width_, int height_, MatType type_) : width(width_), height(height_), type(type_) {
    data = allocData(width, height, type);
    stepCalculate();
}

Mat::Mat(const Mat &M) {
    width = M.width;
    height = M.height;
    type = M.type;
    data = allocData(width, height, type);
    copyData(data, M.data, width, height, type);
    stepCalculate();
}

Mat::Mat(Mat &&M) {
    width = M.width;
    height = M.height;
    type = M.type;
    data = M.data;
    M.data = nullptr;
    stepCalculate();
}

int Mat::getIndex(int width_, int height_) const {
    return height_ * step[0] + width_ * step[1];
}

Mat& Mat::operator=(const Mat &M) {
    width = M.width;
    height = M.height;
    type = M.type;
    delete [] data;
    data = allocData(width, height, type);
    copyData(data, M.data, width, height, type);
    stepCalculate();
    return *this;
}

void Mat::show() {
    
}

