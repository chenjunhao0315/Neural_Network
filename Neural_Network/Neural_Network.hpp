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
#include <unordered_map>

#include "Layer.hpp"

typedef map<string, float> TrainerOption;

class Neural_Network {
public:
    enum nn_status {
        OK,
        ERROR
    };
    Neural_Network(string model_ = "sequential");
    void addLayer(LayerOption opt_);
    void addOutput(string name);
    void makeLayer();
    void shape();
    nn_status status();
    vfloat Forward(Tensor *input_tensor_);
    float Backward(vfloat &target);
    vfloat train(string method, float learning_rate, Tensor *input, vfloat &target_set);
    void train(string method, float learning_rate, vtensor &data_set, vector<vfloat> &target_set, int epoch);
    vfloat predict(Tensor *input);
    float evaluate(vtensor &data_set, vector<vfloat> &target);
    void UpdateNet();
    void ClearGrad();
    bool save(const char *model_name);
    bool load(const char *model_name);
    vector<Tensor*> getDetail();
    vector<vfloat> getDetailParameter();
private:
    string model;
    int layer_number;
    vector<LayerOption> opt_layer;
    vector<Model_Layer> layer;
    vector<string> output_layer;
    unordered_map<string, Tensor*> terminal;
    vector<vector<int>> path;
    int batch_size;
};

class Trainer {
public:
    Trainer(Neural_Network *net, TrainerOption opt);
    vfloat train(Tensor &data, vfloat &target);
    vfloat train(vtensor &data_set, vector<vfloat> &target_set, int epoch);
    void decade(float rate);
    enum Method {
        SGD,
        ADADELTA,
        ADAM
    };
private:
    Neural_Network *network;
    TrainerOption option;
    float learning_rate;
    float l1_decay;
    float l2_decay;
    int batch_size;
    Method method;
    float momentum;
    float ro;
    float eps;
    int iter;
    float beta_1;
    float beta_2;
    vector<float*> gsum;
    vector<float*> xsum;
};

#endif /* Neural_Network_hpp */
