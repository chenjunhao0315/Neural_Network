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
    if (weight)
        delete [] weight;
    if (delta_weight)
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
        for (int i = 0; i < size; ++i) {
            weight[i] = T.weight[i];
            delta_weight[i] = T.delta_weight[i];
        }
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
        size = T.size;
        if (weight)
            delete [] weight;
        weight = new float [size];
        
        if (delta_weight)
            delete [] delta_weight;
        delta_weight = new float [size];
        for (int i = 0; i < size; ++i) {
            weight[i] = T.weight[i];
            delta_weight[i] = T.delta_weight[i];
        }
    }
    return *this;
}

Tensor::Tensor(int width_, int height_, int dimension_) {
    width = width_;
    height = height_;
    dimension = dimension_;
    int n = size = width * height * dimension;
    
    // initialize
    weight = new float [n];
    delta_weight = new float [n];
    fill(weight, weight + n, 0);
    fill(delta_weight, delta_weight + n, 0);
    
    // assign random value
    float scale = sqrt(1.0 / (n));
    for (int i = 0; i < n; ++i) {
        weight[i] = randn(0.0, scale);
    }
}

Tensor::Tensor(int width_, int height_, int dimension_, float parameter) {
    width = width_;
    height = height_;
    dimension = dimension_;
    int n = size = width * height * dimension;
    
    // initialize
    weight = new float [n];
    delta_weight = new float [n];
    for (int i = 0; i < n; ++i) {
        weight[i] = parameter;
    }
    fill(delta_weight, delta_weight + n, 0);
}

Tensor::Tensor(Tensor *T) {
    if (T) {
        width = T->width;
        height = T->height;
        dimension = T->dimension;
        int n = size = width * height * dimension;
        
        // initialize
        weight = new float [n];
        delta_weight = new float [n];
        fill(weight, weight + n, 0);
        fill(delta_weight, delta_weight + n, 0);
        
        for (int i = 0; i < n; ++i) {
            weight[i] = T->weight[i];
        }
    }
}

Tensor::Tensor(vfloat &V) {
    width = 1;
    height = 1;
    dimension = (int)V.size();
    int n = size = width * height * dimension;
    
    // initialize
    weight = new float [n];
    delta_weight = new float [n];
    fill(delta_weight, delta_weight + n, 0);
    
    for (int i = 0; i < n; ++i) {
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
    float *temp = delta_weight;
    delta_weight = new float [size];
    fill(delta_weight, delta_weight + size, 0);
    delete [] temp;
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

void Tensor::set(int width_, int height_, int dimension_, float value) {
    weight[((height_ * width) + width_) * dimension + dimension_] = value;
}

float Tensor::get(int width_, int height_, int dimension_) {
    return weight[((height_ * width) + width_) * dimension + dimension_];
}

float Tensor::getGrad(int width_, int height_, int dimension_) {
    return delta_weight[((height_ * width) + width_) * dimension + dimension_];
}

void Tensor::addGrad(int width_, int height_, int dimension_, float value) {
    if (width_ < 0 || height_ < 0 || width_ >= width || height_ >= height)
        return;
    delta_weight[((width * height_) + width_) * dimension + dimension_] += value;
}
