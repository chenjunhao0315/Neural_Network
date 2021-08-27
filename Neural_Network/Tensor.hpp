//
//  Tensor.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#ifndef Tensor_hpp
#define Tensor_hpp

#include <iostream>
#include <stdio.h>
#include <vector>
#include <cstring>
#include <iomanip>

#include "Utilities.hpp"

using std::fill;
using std::vector;
using std::ostream;

typedef vector<float> vfloat;

class Tensor {
public:
    ~Tensor();
    Tensor();
    Tensor(const Tensor &T);
    Tensor(Tensor &&T);
    Tensor& operator=(const Tensor &T);
    Tensor(Tensor *T);
    Tensor(vfloat &V);
    Tensor(vfloat V1, vfloat V2, vfloat V3, int width_, int height_);
    Tensor(float* RGB, int width_, int height_, int dimension_);
    Tensor(int width_, int height_, int dimension_);
    Tensor(int width_, int height_, int dimension_, float parameter);
    void set(int width_, int height_, int dimension_, float value);
    float get(int width_, int height_, int dimension_);
    float getGrad(int width_, int height_, int dimension_);
    void addGrad(int width_, int height_, int dimension_, float value, int shift_ = 0);
    void showWeight();
    void showDeltaWeight();
    float* getWeight();
    float* getDeltaWeight();
    void clearDeltaWeight();
    int getWidth();
    int getHeight();
    int getDimension();
    int length() {return size;}
    void save(FILE *f);
    void load(FILE *f);
    vfloat toVector();
    void toIMG(const char *filename);
    void shape();
    friend ostream& operator<<(ostream& os, Tensor& t);
//private:
    int width;
    int height;
    int dimension;
    int size;
    float* weight;
    float* delta_weight;
};

#endif /* Tensor_hpp */
