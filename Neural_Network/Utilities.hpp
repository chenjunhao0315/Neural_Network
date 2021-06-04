//
//  Utilities.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#ifndef Utilities_hpp
#define Utilities_hpp

#include <stdio.h>
#include <random>

static std::random_device rd;
static std::default_random_engine generator(rd());
static std::uniform_real_distribution<float> unif(-1.0, 1.0);

float Random();
float Random(float min, float max);
float guassRandom();
float randn(float mu, float std);
float* zero_array(int number);

#endif /* Utilities_hpp */
