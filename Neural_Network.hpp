//
//  Neural_Network.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#ifndef Neural_Network_hpp
#define Neural_Network_hpp


#include <stdio.h>
#include <vector>
#include <random>

#include "Layer.hpp"
#include "Data_Process.hpp"

class Neural_Network {
public:
    void addLayer(LayerOption opt_);
    void makeLayer();
    void shape();
    Tensor* Forward(Tensor *input_tensor_);
    float Backward(float target);
    vfloat train(string method, float learning_rate, Tensor *input, float &target);
    void train(string method, float learning_rate, vtensor &data_set, vfloat &target, int epoch);
    vfloat predict(Tensor *input);
private:
    vector<LayerOption> opt_layer;
    vector<Model_Layer> layer;
};

#endif /* Neural_Network_hpp */