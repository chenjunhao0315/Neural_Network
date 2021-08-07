//
//  Mat.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/8/5.
//

#include "Mat.hpp"

// Mat
Mat::~Mat() {
    freeData();
}

void Mat::freeData() {
    delete [] data;
}

void Mat::copyFrom(void *src) {
    memcpy(data, src, step[0] * height);
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
        default:
            return nullptr;
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
        default:
            break;
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
        default:
            step[1] = 0;
    }
    step[0] = width * step[1];
}

int Mat::depth() {
    switch (type) {
        case MAT_8UC1:
        case MAT_8SC1:
        case MAT_32UC1:
        case MAT_32SC1:
        case MAT_32FC1:
            return 1; break;
        case MAT_8UC3:
        case MAT_8SC3:
        case MAT_32UC3:
        case MAT_32SC3:
        case MAT_32FC3:
            return 3; break;
        default:
            return -1;
    }
    return -1;
}

Mat::Mat(int width_, int height_, MatType type_, Scalar scalar) : width(width_), height(height_), type(type_) {
    data = allocData(width, height, type);
    stepCalculate();
    if (!(scalar.val[0] == 0 && scalar.val[1] == 0 && scalar.val[2] == 0)) {
        fillWith(scalar);
    }
}

Mat::Mat(int width_, int height_, MatType type_, void *src) : width(width_), height(height_), type(type_) {
    data = allocData(width, height, type);
    stepCalculate();
    copyFrom(src);
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

Mat& Mat::operator=(std::initializer_list<float> list) {
    unsigned char *dst_ptr = data;
    int elemSize1 = step[1] / depth();
    double buf[3];
    for (float element : list) {
        Scalar s(element);
        convertScalar(s, buf, type);
        for (int i = 0; i < elemSize1; ++i) {
            *(dst_ptr + i) = *((unsigned char*)buf + i);
        }
        dst_ptr += elemSize1;
    }
    return *this;
}

Mat Mat::convertTo(MatType dstType, float scale, float shift) {
    Mat dst(width, height, dstType);
    func cvtFunc = getConvertFunc(type, dstType);
    if (!cvtFunc) {
        printf("[Mat][Convert] Channel size unmatched!\n");
        return Mat();
    }
    if (dstType % 2)
        cvtFunc(data, dst.ptr(), 3 * width * height);
    else
        cvtFunc(data, dst.ptr(), width * height);
    return dst;
}

func getConvertFunc(MatType srcType, MatType dstType) {
    static func cvtTable[5][5] = {
        {convertType<unsigned char, unsigned char>, convertType<unsigned char, char>, convertType<unsigned char, unsigned int>, convertType<unsigned char, int>, convertType<unsigned char, float>},
        {convertType<char, unsigned char>, convertType<char, char>, convertType<char, unsigned int>, convertType<char, int>, convertType<char, float>},
        {convertType<unsigned int, unsigned char>, convertType<unsigned int, char>, convertType<unsigned int, unsigned int>, convertType<unsigned int, int>, convertType<unsigned int, float>},
        {convertType<int, unsigned char>, convertType<int, char>, convertType<int, unsigned int>, convertType<int, int>, convertType<int, float>},
        {convertType<float, unsigned char>, convertType<float, char>, convertType<float, unsigned int>, convertType<float, int>, convertType<float, float>}
    };
    if (srcType % 2 != dstType % 2) {
        return nullptr;
    }
    return cvtTable[srcType >> 1][dstType >> 1];
}

void Mat::fillWith(Scalar &s) {
    double buf[3];
    convertScalar(s, buf, type);
    unsigned char *dst = data;
    unsigned char *src = (unsigned char *)buf;
    int size = width * height;
    for (int i = size; i--; ) {
        for (int j = 0; j < step[1]; ++j) {
            *(dst++) = *(src + j);
        }
    }
}
// End Mat

// ConvTool
void ConvTool::start(Mat &src, Mat &dst) {
    if (kernel.depth() != 1) {
        printf("[Mat][ConvTool] Only accept 1 channel kernel!\n");
        return;
    }
    kernel = kernel.convertTo(MAT_32FC1);
    
    if (srcType == MAT_8UC1 || srcType == MAT_8UC3) {
        if (dstType == MAT_8UC1 || dstType == MAT_8UC3) {
            ConvEngine<unsigned char, unsigned char> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_8SC1 || dstType == MAT_8SC3) {
            ConvEngine<unsigned char, char> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_32UC1 || dstType == MAT_32UC3) {
            ConvEngine<unsigned char, unsigned int> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_32SC1 || dstType == MAT_32SC3) {
            ConvEngine<unsigned char, int> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_32FC1 || dstType == MAT_32FC3) {
            ConvEngine<unsigned char, float> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        }
    } else if (srcType == MAT_8SC1 || srcType == MAT_8SC3) {
        if (dstType == MAT_8UC1 || dstType == MAT_8UC3) {
            ConvEngine<char, unsigned char> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_8SC1 || dstType == MAT_8SC3) {
            ConvEngine<char, char> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_32UC1 || dstType == MAT_32UC3) {
            ConvEngine<char, unsigned int> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_32SC1 || dstType == MAT_32SC3) {
            ConvEngine<char, int> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_32FC1 || dstType == MAT_32FC3) {
            ConvEngine<char, float> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        }
    } else if (srcType == MAT_32UC1 || srcType == MAT_32UC3) {
        if (dstType == MAT_8UC1 || dstType == MAT_8UC3) {
            ConvEngine<unsigned int, unsigned char> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_8SC1 || dstType == MAT_8SC3) {
            ConvEngine<unsigned int, char> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_32UC1 || dstType == MAT_32UC3) {
            ConvEngine<unsigned int, unsigned int> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_32SC1 || dstType == MAT_32SC3) {
            ConvEngine<unsigned int, int> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_32FC1 || dstType == MAT_32FC3) {
            ConvEngine<unsigned int, float> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        }
    } else if (srcType == MAT_32SC1 || srcType == MAT_32SC3) {
        if (dstType == MAT_8UC1 || dstType == MAT_8UC3) {
            ConvEngine<int, unsigned char> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_8SC1 || dstType == MAT_8SC3) {
            ConvEngine<int, char> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_32UC1 || dstType == MAT_32UC3) {
            ConvEngine<int, unsigned int> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_32SC1 || dstType == MAT_32SC3) {
            ConvEngine<int, int> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_32FC1 || dstType == MAT_32FC3) {
            ConvEngine<int, float> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        }
    } else if (srcType == MAT_32FC1 || srcType == MAT_32FC3) {
        if (dstType == MAT_8UC1 || dstType == MAT_8UC3) {
            ConvEngine<float, unsigned char> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_8SC1 || dstType == MAT_8SC3) {
            ConvEngine<float, char> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_32UC1 || dstType == MAT_32UC3) {
            ConvEngine<float, unsigned int> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_32SC1 || dstType == MAT_32SC3) {
            ConvEngine<float, int> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        } else if (dstType == MAT_32FC1 || dstType == MAT_32FC3) {
            ConvEngine<float, float> engine(dst.depth(), dst.width, dst.height, src.depth(), src.width, kernel.width);
            engine.process(src.ptr(), dst.ptr(), kernel.ptr());
        }
    }
}
// End ConvTool

// Scalar
Scalar::Scalar(float v0, float v1, float v2) {
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
}

void convertScalar(Scalar &s, const void *dst, MatType type) {
    switch (type) {
        case MAT_8UC1: {
            unsigned char *buf = (unsigned char *)dst;
            buf[0] = (unsigned char)s.val[0];
            break;
        }
        case MAT_8UC3: {
            unsigned char *buf = (unsigned char *)dst;
            buf[0] = (unsigned char)s.val[0];
            buf[1] = (unsigned char)s.val[1];
            buf[2] = (unsigned char)s.val[2];
            break;
        }
        case MAT_8SC1: {
            char *buf = (char *)dst;
            buf[0] = (char)s.val[0];
            break;
        }
        case MAT_8SC3: {
            char *buf = (char *)dst;
            buf[0] = (char)s.val[0];
            buf[1] = (char)s.val[1];
            buf[2] = (char)s.val[2];
            break;
        }
        case MAT_32UC1: {
            unsigned int *buf = (unsigned int *)dst;
            buf[0] = (unsigned int)s.val[0];
            break;
        }
        case MAT_32UC3: {
            unsigned int *buf = (unsigned int *)dst;
            buf[0] = (unsigned int)s.val[0];
            buf[1] = (unsigned int)s.val[1];
            buf[2] = (unsigned int)s.val[2];
            break;
        }
        case MAT_32SC1: {
            int *buf = (int *)dst;
            buf[0] = (int)s.val[0];
            break;
        }
        case MAT_32SC3: {
            int *buf = (int *)dst;
            buf[0] = (int)s.val[0];
            buf[1] = (int)s.val[1];
            buf[2] = (int)s.val[2];
            break;
        }
        case MAT_32FC1: {
            float *buf = (float *)dst;
            buf[0] = (float)s.val[0];
            break;
        }
        case MAT_32FC3: {
            float *buf = (float *)dst;
            buf[0] = (float)s.val[0];
            buf[1] = (float)s.val[1];
            buf[2] = (float)s.val[2];
            break;
        }
        default:
            break;
    }
}
// End Scalar
