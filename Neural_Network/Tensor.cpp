//
//  Tensor.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include "Tensor.hpp"

Tensor::Tensor() {
    width = 0;
    height = 0;
    dimension = 0;
    size = 0;
    weight = nullptr;
    delta_weight = nullptr;
}

Tensor::~Tensor() {
    delete [] weight;
    delete [] delta_weight;
}

Tensor::Tensor(const Tensor &T) {
    if (this != &T) {
        width = T.width;
        height = T.height;
        dimension = T.dimension;
        size = T.size;
        weight = new float [size];
        delta_weight = new float [size];
        int copy_size = sizeof(float) * size;
        memcpy(weight, T.weight, copy_size);
        memcpy(delta_weight, T.delta_weight, copy_size);
    }
}

Tensor::Tensor(Tensor &&T) {
    width = T.width;
    height = T.height;
    dimension = T.dimension;
    size = T.size;
    weight = T.weight;
    T.weight = nullptr;
    delta_weight = T.delta_weight;
    T.delta_weight = nullptr;
}

Tensor& Tensor::operator=(const Tensor &T) {
    if (this != &T) {
        width = T.width;
        height = T.height;
        dimension = T.dimension;
        if (size != T.size) {
            delete [] weight;
            weight = new float [T.size];
            delete [] delta_weight;
            delta_weight = new float [T.size];
        }
        size = T.size;
        memcpy(weight, T.weight, sizeof(float) * size);
        memcpy(delta_weight, T.delta_weight, sizeof(float) * size);
    }
    return *this;
}

Tensor::Tensor(int width_, int height_, int dimension_) {
    width = width_;
    height = height_;
    dimension = dimension_;
    size = width * height * dimension;
    
    // initialize
    weight = new float [size];
    delta_weight = new float [size]();
    
    // assign random value
    float scale = sqrt(1.0 / size);
    for (int i = size; --i; ) {
        weight[i] = randn(0.0, scale);
    }
}

Tensor::Tensor(int width_, int height_, int dimension_, float parameter) {
    width = width_;
    height = height_;
    dimension = dimension_;
    size = width * height * dimension;
    
    // initialize
    weight = new float [size];
    delta_weight = new float [size]();
    fill(weight, weight + size, parameter);
}

Tensor::Tensor(Tensor *T) {
    if (T) {
        width = T->width;
        height = T->height;
        dimension = T->dimension;
        size = width * height * dimension;
        
        // initialize
        weight = new float [size]();
        delta_weight = new float [size]();
        //fill(weight, weight + size, 0);
//        fill(delta_weight, delta_weight + size, 0);
        
        //float *weight_src = T->weight;
        //float *weight_dst = weight;
        memcpy(weight, T->weight, sizeof(float) * size);
        /*for (int i = 0; i < size; ++i) {
            *weight_dst++ = *weight_src++;
        }*/
    }
}

Tensor::Tensor(vfloat &V) {
    width = 1;
    height = 1;
    dimension = (int)V.size();
    size = width * height * dimension;
    
    // initialize
    weight = new float [size];
    delta_weight = new float [size]();
    //fill(delta_weight, delta_weight + n, 0);
    for (int i = 0; i < size; ++i) {
        weight[i] = V[i];
    }
}

Tensor::Tensor(vfloat V1, vfloat V2, vfloat V3, int width_, int height_) {
    width = width_;
    height = height_;
    dimension = 3;
    int n = size = width * height;
    size *= 3;
    
    // initialize
    weight = new float [n * 3];
    delta_weight = new float [n * 3];
    fill(weight, weight + n, 0);
    fill(delta_weight, delta_weight + n, 0);
    
    for (int i = 0; i < n; ++i) {
        weight[i * 3] = V1[i];
        weight[i * 3 + 1] = V2[i];
        weight[i * 3 + 2] = V3[i];
    }
}

Tensor::Tensor(float* pixelArray, int width_, int height_, int dimension_) {
    width = width_;
    height = height_;
    dimension = dimension_;
    int n = size = width * height * dimension;
    
    // initialize
    weight = new float [n];
    delta_weight = new float [n];
    fill(weight, weight + n, 0);
    fill(delta_weight, delta_weight + n, 0);
    
    for (int i = 0; i < n; ++i) {
        weight[i] = pixelArray[i];
    }
}

float* Tensor::getWeight() {
    return weight;
}

float* Tensor::getDeltaWeight() {
    return delta_weight;
}

void Tensor::showWeight() {
    int n = size;
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", weight[i]);
    }
    printf("\n");
}

void Tensor::showDeltaWeight() {
    int n = size;
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", delta_weight[i]);
    }
    printf("\n");
}

void Tensor::clearDeltaWeight() {
    fill(delta_weight, delta_weight + size, 0);
}

int Tensor::getWidth() {
    return width;
}

int Tensor::getHeight() {
    return height;
}

int Tensor::getDimension() {
    return dimension;
}

void Tensor::shape() {
    printf("width: %d height: %d dimension: %d size: %d\n", width, height, dimension, size);
}

void Tensor::set(int width_, int height_, int dimension_, float value) {
    weight[((height_ * width) + width_) * dimension + dimension_] = value;
}

float Tensor::get(int width_, int height_, int dimension_) {
    return weight[((height_ * width) + width_) * dimension + dimension_];
}

float Tensor::getGrad(int width_, int height_, int dimension_) {
    return delta_weight[((height_ * width) + width_) * dimension + dimension_];
}

void Tensor::addGrad(int width_, int height_, int dimension_, float value, int shift_, int batch) {
    delta_weight[((width * height_) + width_) * dimension / batch + dimension_ + shift_] += value;
}

void Tensor::save(FILE *f) {
    fwrite(&width, sizeof(int), 1, f);
    fwrite(&height, sizeof(int), 1, f);
    fwrite(&dimension, sizeof(int), 1, f);
    fwrite(&size, sizeof(int), 1, f);
    fwrite(weight, sizeof(float), size, f);
}

void Tensor::load(FILE *f) {
    fread(&width, sizeof(int), 1, f);
    fread(&height, sizeof(int), 1, f);
    fread(&dimension, sizeof(int), 1, f);
    fread(&size, sizeof(int), 1, f);
    delete [] weight;
    weight = new float [size];
    fread(weight, sizeof(float), size, f);
}

vfloat Tensor::toVector() {
    vfloat result;
    result.reserve(size);
    float *weight_ptr = weight;
    for (int i = size; i; --i) {
        result.push_back(*(weight_ptr++));
    }
    return result;
}

void Tensor::toIMG(const char *filename) {
    FILE *f = fopen(filename, "wb");
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    unsigned char *pixel = new unsigned char [3 * width * height];
    for (int i = 0; i < 3 * width * height; ++i) {
        pixel[i] = (unsigned char)(weight[i] * 255);
    }
    fwrite(pixel, sizeof(unsigned char), 3 * width * height, f);
    fclose(f);
}

ostream& operator<<(ostream& os, Tensor& t) {
    for (int d = 0; d < t.dimension; ++d) {
        os << "Dim: " << d << std::endl;
        for (int h = 0; h < t.height; ++h) {
            for (int w = 0; w < t.width; ++w) {
                os << std::fixed << std::setprecision(2) << (t.get(w, h, d)) << " ";
            }
            os << std::endl;
        }
    }
    return os;
}
