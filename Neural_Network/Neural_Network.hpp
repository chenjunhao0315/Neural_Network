//
//  Neural_Network.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#ifndef Neural_Network_hpp
#define Neural_Network_hpp


#include <iostream>
#include <stdio.h>
#include <vector>
#include <random>
#include <algorithm>

#include "Layer.hpp"
#include "Data_Process.hpp"

class Neural_Network {
public:
    Neural_Network(string model_ = "sequential");
    void addLayer(LayerOption opt_);
    void addOutput(string name);
    void makeLayer();
    void shape();
    vfloat Forward(Tensor *input_tensor_);
    float Backward(vfloat &target);
    vfloat train(string method, float learning_rate, Tensor *input, vfloat &target_set);
    void train(string method, float learning_rate, vtensor &data_set, vector<vfloat> &target_set, int epoch);
    vfloat predict(Tensor *input);
    float evaluate(vtensor &data_set, vector<vfloat> &target);
    bool save(const char *model_name);
    bool load(const char *model_name);
private:
    string model;
    int layer_number;
    vector<LayerOption> opt_layer;
    vector<Model_Layer> layer;
    vector<string> output_layer;
    map<string, Tensor*> terminal;
};

#endif /* Neural_Network_hpp */
