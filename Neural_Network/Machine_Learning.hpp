//
//  Machine_Learning.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/10/20.
//

#ifndef Machine_Learning_hpp
#define Machine_Learning_hpp

#include <stdio.h>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <cmath>

using namespace std;

typedef vector<int> vint;
typedef vector<float> vfloat;

class StandardScaler {
public:
    StandardScaler() {}
    void fit(vector<vfloat> &data);
    vector<vfloat> transform(vector<vfloat> &data);
private:
    vfloat means;
    vfloat stdevs;
};

class GaussianNB {
public:
    GaussianNB() {}
    void fit(vector<vfloat> &data, vint &label);
    unordered_map<int, vector<vfloat>> separate_classes(vector<vfloat> &data, vint &label);
    void summarize(unordered_map<int, vector<vfloat>> &X);
    float gauss_distribution_function(float x, float mean, float stdev);
    int predict(vfloat &data);
    vint predict(vector<vfloat> &data);
private:
    unordered_map<int, tuple<float, vfloat, vfloat>> summary;
};

float accuracy_score(vint &pred, vint &truth);

#endif /* Machine_Learning_hpp */
