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
    Tensor(vfloat &V);
    Tensor(vfloat V1, vfloat V2, vfloat V3, int width_, int height_);
    Tensor(float* RGB, int width_, int height_, int channel_);
    Tensor(int batch_, int channel_, int height_, int width_);
    Tensor(int batch_, int channel_, int height_, int width_, float parameter);
    void reshape(int batch_, int channel_, int height_, int width_, bool extend = false);
    void free();
    void extend();
    void copyTo(Tensor &T);
    void one_of_n_encodinig(int index, int n);
    Tensor concate(const Tensor &T);
    float get(int batch_, int channel_, int height_, int width_);
    void showWeight();
    void showDeltaWeight();
    float* getWeight();
    float* getDeltaWeight();
    void clearWeight();
    void clearDeltaWeight();
    int getWidth();
    int getHeight();
    int getDimension();
    int length() {return size;}
    void save_raw(FILE *f);
    void load_raw(FILE *f);
    vfloat toVector();
    void toIMG(const char *filename);
    void shape();
    friend ostream& operator<<(ostream& os, Tensor& t);
//private:
    int batch;
    int channel;
    int height;
    int width;
    int size;
    float* weight;
    float* delta_weight;
};

bool same_plane(const Tensor &a, const Tensor &b);
bool same_block(const Tensor &a, const Tensor &b);
bool same_structure(const Tensor &a, const Tensor &b);

Tensor operator+(const Tensor &a, const Tensor &b);
Tensor operator-(const Tensor &a, const Tensor &b);

#endif /* Tensor_hpp */
