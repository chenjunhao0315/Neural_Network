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
#include <cassert>
#include <iomanip>
#include <initializer_list>

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
    Tensor& operator=(float c);
    Tensor& operator=(std::initializer_list<float> list);
    bool operator==(const Tensor &T) const;
    Tensor& operator+=(const Tensor &T);
    Tensor& operator-=(const Tensor &T);
    float& operator[](int index);
    const float& operator[](int index) const;
    Tensor(Tensor *T);
    Tensor(vfloat &V);
    Tensor(vfloat V1, vfloat V2, vfloat V3, int width_, int height_);
    Tensor(float* RGB, int width_, int height_, int dimension_);
    Tensor(int width_, int height_, int dimension_);
    Tensor(int width_, int height_, int dimension_, float parameter);
    void extend();
    void copyTo(Tensor &T);
    Tensor concate(const Tensor &T);
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
    void save_raw(FILE *f);
    void load(FILE *f);
    void load_raw(FILE *f);
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

bool same_plane(const Tensor &a, const Tensor &b);
bool same_structure(const Tensor &a, const Tensor &b);

Tensor operator+(const Tensor &a, const Tensor &b);
Tensor operator-(const Tensor &a, const Tensor &b);

#endif /* Tensor_hpp */
