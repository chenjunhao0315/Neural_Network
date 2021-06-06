//
//  Tensor.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include "Tensor.hpp"

Tensor::Tensor() {
    weight = nullptr;
    delta_weight = nullptr;
}

Tensor::~Tensor() {
//    std::cout << "Free " << weight << " " << delta_weight << std::endl;
    if (weight)
        delete [] weight;
    if (delta_weight)
        delete [] delta_weight;
}

void Tensor::operator=(const Tensor &T) {
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

Tensor::Tensor(int width_, int height_, int dimension_) {
    width = width_;
    height = height_;
    dimension = dimension_;
    int n = size = width * height * dimension;
    
    // initialize
    //weight.assign(n, 0);
    //delta_weight.assign(n, 0);
    weight = new float [n];
    delta_weight = new float [n];
    fill(weight, weight + n, 0);
    fill(delta_weight, delta_weight + n, 0);
//    memset(weight, 0, n * sizeof(float));
//    memset(delta_weight, 0, n * sizeof(float));
    
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
    //weight.assign(n, parameter);
    //delta_weight.assign(n, 0);
    weight = new float [n];
    delta_weight = new float [n];
    for (int i = 0; i < n; ++i) {
        weight[i] = parameter;
    }
//    memset(weight, parameter, n * sizeof(float));
    fill(delta_weight, delta_weight + n, 0);
}

Tensor::Tensor(Tensor *T) {
    width = T->width;
    height = T->height;
    dimension = T->dimension;
    int n = size = width * height * dimension;
    
    // initialize
    //weight.assign(n, 0);
    //delta_weight.assign(n, 0);
    weight = new float [n];
    delta_weight = new float [n];
    fill(weight, weight + n, 0);
    fill(delta_weight, delta_weight + n, 0);
//    memset(weight, 0, n * sizeof(float));
//    memset(delta_weight, 0, n * sizeof(float));
    
    for (int i = 0; i < n; ++i) {
        weight[i] = T->weight[i];
    }
}

Tensor::Tensor(vfloat V) {
    width = 1;
    height = 1;
    dimension = (int)V.size();
    int n = size = width * height * dimension;
    
    // initialize
    //weight.assign(n, 0);
    //delta_weight.assign(n, 0);
    weight = new float [n];
    delta_weight = new float [n];
//    memset(weight, 0, n * sizeof(float));
//    memset(delta_weight, 0, n * sizeof(float));
    fill(weight, weight + n, 0);
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
//    memset(weight, 0, n * 3 * sizeof(float));
//    memset(delta_weight, 0, n * 3 * sizeof(float));
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
//    memset(delta_weight, 0, sizeof(float) * size);
    delete [] temp;
    //delta_weight.assign(delta_weight.size(), 0);
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
