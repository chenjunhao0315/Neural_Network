//
//  Tensor.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include "Tensor.hpp"

Tensor::Tensor(int width_, int height_, int dimension_) {
    width = width_;
    height = height_;
    dimension = dimension_;
    int n = width * height * dimension;
    
    // initialize
    weight.assign(n, 0);
    delta_weight.assign(n, 0);
    
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
    int n = width * height * dimension;
    
    // initialize
    weight.assign(n, parameter);
    delta_weight.assign(n, 0);
}

Tensor::Tensor(Tensor *T) {
    width = T->width;
    height = T->height;
    dimension = T->dimension;
    int n = width * height * dimension;
    
    // initialize
    weight.assign(n, 0);
    delta_weight.assign(n, 0);
    
    for (int i = 0; i < n; ++i) {
        weight[i] = T->weight[i];
    }
}

Tensor::Tensor(vfloat V) {
    width = 1;
    height = 1;
    dimension = (int)V.size();
    int n = width * height * dimension;
    
    // initialize
    weight.assign(n, 0);
    delta_weight.assign(n, 0);
    
    for (int i = 0; i < n; ++i) {
        weight[i] = V[i];
    }
}

Tensor::Tensor(vfloat V1, vfloat V2, vfloat V3, int width_, int height_) {
    width = width_;
    height = height_;
    dimension = 3;
    int n = width * height;
    
    // initialize
    weight.assign(n * 3, 0);
    delta_weight.assign(n * 3, 0);
    
    for (int i = 0; i < n; ++i) {
        weight[i * 3] = V1[i];
        weight[i * 3 + 1] = V2[i];
        weight[i * 3 + 2] = V3[i];
    }
}

vector<float>& Tensor::getWeight() {
    return weight;
}

vector<float>& Tensor::getDeltaWeight() {
    return delta_weight;
}

void Tensor::showWeight() {
    int n = (int)weight.size();
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", weight[i]);
    }
    printf("\n");
}

void Tensor::showDeltaWeight() {
    int n = (int)delta_weight.size();
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", delta_weight[i]);
    }
    printf("\n");
}

void Tensor::clearDeltaWeight() {
    delta_weight.assign(delta_weight.size(), 0);
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

float Tensor::getGrad(int width_, int height_, int dimension_) {
    return delta_weight[((height_ * width) + width_) * dimension + dimension_];
}
