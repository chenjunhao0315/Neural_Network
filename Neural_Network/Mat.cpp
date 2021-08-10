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
    data = nullptr;
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

void Mat::release() {
    freeData();
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
    if (depth() != getDepth(dstType)) {
        printf("[Mat][Convert] Channel size unmatched!\n");
        return Mat();
    }
    BinaryFunc cvtFunc = getConvertTypeFunc(type, dstType);
    if (!cvtFunc) {
        return *this;
    } else {
        if (getDepth(dstType) == 3)
            cvtFunc(data, dst.ptr(), 3 * width * height);
        else
            cvtFunc(data, dst.ptr(), width * height);
    }
    return dst;
}

Mat Mat::subtract(Mat &minuend_, MatType dstType_) {
    if (depth() != minuend_.depth()) {
        printf("[Mat][Subtract] Channel unmatched!\n");
        return Mat();
    }
    if (!(width == minuend_.width && height == minuend_.height)) {
        printf("[Mat][Subtract] Size unmatched!\n");
        return Mat();
    }
    
    
    MatType dstType = (dstType_ == MAT_UNDEFINED) ? type : dstType_;
    Mat minuend = minuend_.convertTo(dstType);
    Mat dst(width, height, dstType);
    
    TernaryFunc subFunc = getsubtractMatFunc(type, dstType);
    if (getDepth(dstType) == 3)
        subFunc(data, minuend.ptr(), dst.ptr(), width * height * 3);
    else
        subFunc(data, minuend.ptr(), dst.ptr(), width * height);
    return dst;
}

Mat Mat::add(Mat &addend_, MatType dstType_) {
    if (depth() != addend_.depth()) {
        printf("[Mat][Add] Channel unmatched!\n");
        return Mat();
    }
    if (!(width == addend_.width && height == addend_.height)) {
        printf("[Mat][Add] Size unmatched!\n");
        return Mat();
    }
    
    
    MatType dstType = (dstType_ == MAT_UNDEFINED) ? type : dstType_;
    Mat addend = addend_.convertTo(dstType);
    Mat dst(width, height, dstType);
    
    TernaryFunc addFunc = getaddMatFunc(type, dstType);
    if (getDepth(dstType) == 3)
        addFunc(data, addend.ptr(), dst.ptr(), width * height * 3);
    else
        addFunc(data, addend.ptr(), dst.ptr(), width * height);
    return dst;
}

Mat Mat::divide(Mat &dividend_, MatType dstType_) {
    if (depth() != dividend_.depth()) {
        printf("[Mat][Divide] Channel unmatched!\n");
        return Mat();
    }
    if (!(width == dividend_.width && height == dividend_.height)) {
        printf("[Mat][Divide] Size unmatched!\n");
        return Mat();
    }
    
    
    MatType dstType = (dstType_ == MAT_UNDEFINED) ? type : dstType_;
    Mat dividend = dividend_.convertTo(dstType);
    Mat dst(width, height, dstType);
    
    TernaryFunc divideFunc = getdivideMatFunc(type, dstType);
    if (getDepth(dstType) == 3)
        divideFunc(data, dividend.ptr(), dst.ptr(), width * height * 3);
    else
        divideFunc(data, dividend.ptr(), dst.ptr(), width * height);
    return dst;
}

Mat Mat::addWeighted(float alpha, Mat &addend, float beta, float gamma, MatType dstType) {
    
    Mat src1(width, height, type);
    Mat gain_alpha(1, 1, MAT_32FC1, Scalar(alpha));
    ConvTool stage_1(type, dstType, gain_alpha);
    stage_1.start(*this, src1);
    Mat src2(width, height, type);
    Mat gain_beta(1, 1, MAT_32FC1, Scalar(beta));
    ConvTool stage_2(addend.getType(), dstType, gain_beta);
    stage_2.start(addend, src2);
    
    Mat dst = src1.add(src2, dstType);
    Mat gain_gamma(width, height, dstType, Scalar(gamma, gamma, gamma));
    dst = dst.add(gain_gamma);
    
    return dst;
}

Mat Mat::absScale(float scale, float alpha) {
    Mat dst(width, height, type);
    
    if (scale != 1) {
        Mat gain(1, 1, MAT_32FC1, Scalar(scale));
        ConvTool Amplifier(type, type, gain);
        Amplifier.start(*this, dst);
    }
    if (alpha != 0) {
        MatType offset_type = (depth() == 1) ? MAT_32FC1 : MAT_32FC3;
        Mat offset(width, height, offset_type, Scalar(alpha, alpha, alpha));
        dst.add(offset);
    }

    if (type == MAT_8SC1 || type == MAT_8SC3 || type == MAT_32SC1 || type == MAT_32SC3 || type == MAT_32FC1 || type == MAT_32FC3) {
        BinaryFunc absFunc = getabsMatFunc(type);
        if (depth() == 1) {
            absFunc(data, dst.ptr(), width * height);
        } else {
            absFunc(data, dst.ptr(), width * height * 3);
        }
    }
    return dst;
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

BinaryFunc getConvertTypeFunc(MatType srcType, MatType dstType) {
    static BinaryFunc cvtTable[5][5] = {
        {nullptr, convertType<unsigned char, char>, convertType<unsigned char, unsigned int>, convertType<unsigned char, int>, convertType<unsigned char, float>},
        {convertType<char, unsigned char>, nullptr, convertType<char, unsigned int>, convertType<char, int>, convertType<char, float>},
        {convertType<unsigned int, unsigned char>, convertType<unsigned int, char>, nullptr, convertType<unsigned int, int>, convertType<unsigned int, float>},
        {convertType<int, unsigned char>, convertType<int, char>, convertType<int, unsigned int>, nullptr, convertType<int, float>},
        {convertType<float, unsigned char>, convertType<float, char>, convertType<float, unsigned int>, convertType<float, int>, nullptr}
    };
    if (srcType % 2 != dstType % 2) {
        return nullptr;
    }
    return cvtTable[srcType >> 1][dstType >> 1];
}

BinaryFunc getabsMatFunc(MatType type) {
    switch (type) {
        case MAT_8SC1:
        case MAT_8SC3:
            return absMat<char>; break;
        case MAT_32SC1:
        case MAT_32SC3:
            return absMat<int>; break;
        case MAT_32FC1:
        case MAT_32FC3:
            return absMat<float>; break;
        default:
            break;
    }
    return nullptr;
}

TernaryFunc getsubtractMatFunc(MatType srcType, MatType dstType) {
    static TernaryFunc subTable[5][5] = {
        {subtractMat<unsigned char, unsigned char>, subtractMat<unsigned char, char>, subtractMat<unsigned char, unsigned int>, subtractMat<unsigned char, int>, subtractMat<unsigned char, float>},
        {subtractMat<char, unsigned char>, subtractMat<char, char>, subtractMat<char, unsigned int>, subtractMat<char, int>, subtractMat<char, float>},
        {subtractMat<unsigned int, unsigned char>, subtractMat<unsigned int, char>, subtractMat<unsigned int, unsigned int>, subtractMat<unsigned int, int>, subtractMat<unsigned int, float>},
        {subtractMat<int, unsigned char>, subtractMat<int, char>, subtractMat<int, unsigned int>, subtractMat<int, int>, subtractMat<int, float>},
        {subtractMat<float, unsigned char>, subtractMat<float, char>, subtractMat<float, unsigned int>, subtractMat<float, int>, subtractMat<float, float>}
    };
    
    return subTable[srcType >> 1][dstType >> 1];
}

TernaryFunc getaddMatFunc(MatType srcType, MatType dstType) {
    static TernaryFunc addTable[5][5] = {
        {addMat<unsigned char, unsigned char>, addMat<unsigned char, char>, addMat<unsigned char, unsigned int>, addMat<unsigned char, int>, addMat<unsigned char, float>},
        {addMat<char, unsigned char>, addMat<char, char>, addMat<char, unsigned int>, addMat<char, int>, addMat<char, float>},
        {addMat<unsigned int, unsigned char>, addMat<unsigned int, char>, addMat<unsigned int, unsigned int>, addMat<unsigned int, int>, addMat<unsigned int, float>},
        {addMat<int, unsigned char>, addMat<int, char>, addMat<int, unsigned int>, addMat<int, int>, addMat<int, float>},
        {addMat<float, unsigned char>, addMat<float, char>, addMat<float, unsigned int>, addMat<float, int>, addMat<float, float>}
    };
    
    return addTable[srcType >> 1][dstType >> 1];
}

TernaryFunc getdivideMatFunc(MatType srcType, MatType dstType) {
    static TernaryFunc addTable[5][5] = {
        {divideMat<unsigned char, unsigned char>, divideMat<unsigned char, char>, divideMat<unsigned char, unsigned int>, divideMat<unsigned char, int>, divideMat<unsigned char, float>},
        {divideMat<char, unsigned char>, divideMat<char, char>, divideMat<char, unsigned int>, divideMat<char, int>, divideMat<char, float>},
        {divideMat<unsigned int, unsigned char>, divideMat<unsigned int, char>, divideMat<unsigned int, unsigned int>, divideMat<unsigned int, int>, divideMat<unsigned int, float>},
        {divideMat<int, unsigned char>, divideMat<int, char>, divideMat<int, unsigned int>, divideMat<int, int>, divideMat<int, float>},
        {divideMat<float, unsigned char>, divideMat<float, char>, divideMat<float, unsigned int>, divideMat<float, int>, divideMat<float, float>}
    };
    
    return addTable[srcType >> 1][dstType >> 1];
}

int getDepth(MatType type) {
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
// End Mat

// ConvTool
void ConvTool::start(Mat &src, Mat &dst) {
    if (kernel.depth() != 1) {
        printf("[Mat][ConvTool] Only accept 1 channel kernel!\n");
        return;
    }
    if (kernel.width != kernel.height) {
        printf("[Mat][ConvTool] Only accept square kernel!\n");
        return;
    }
    if (!(kernel.width % 2 && kernel.height % 2)) {
        printf("[Mat][ConvTool] Only odd size kernel!\n");
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
