//
//  Tensor.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#ifndef Tensor_hpp
#define Tensor_hpp

#include <stdio.h>
#include <vector>

#include "Utilities.hpp"

using std::vector;

typedef vector<float> vfloat;

class Tensor {
public:
    Tensor() {}
    Tensor(Tensor *T);
    Tensor(vfloat V);
    Tensor(vfloat V1, vfloat V2, vfloat V3, int width_, int height_);
    Tensor(int width_, int height_, int dimension_);
    Tensor(int width_, int height_, int dimension_, float parameter);
    void set(int width_, int height_, int dimension_, float value);
    float getGrad(int width_, int height_, int dimension_);
    void showWeight();
    void showDeltaWeight();
    vfloat& getWeight();
    vfloat& getDeltaWeight();
    void clearDeltaWeight();
    int getWidth();
    int getHeight();
    int getDimension();
private:
    int width;
    int height;
    int dimension;
    vfloat weight;
    vfloat delta_weight;
};

#endif /* Tensor_hpp */
