//
//  Utilities.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include "Utilities.hpp"

static bool return_v = false;
static float v_val = 0.0;

float randn(float mu, float std) {
    return mu + guassRandom() * std;
}

float guassRandom() {
    if (return_v) {
        return_v = false;
        return v_val;
    }
    float u = Random();
    float v = Random();
    float r = u * u + v * v;
    if (r == 0 || r > 1)
        return guassRandom();
    float c = sqrt(-2 * log(r) / r);
    v_val = v * c;
    return_v = true;
    return u * c;
}

float Random() {
    return unif(generator);
}
