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
    ~Neural_Network() {delete [] layer; delete [] workspace;}
    Neural_Network(string model_ = "sequential");
    void addLayer(LayerOption opt_);
    void addOutput(string name);
    void compile(int batch_size_ = 1);
    void shape();
    int getBatchSize() {return batch_size;}
    nn_status status();
    vtensorptr Forward(Tensor *input_tensor_, bool train = false);
    float Backward(vfloat &target);
    vfloat predict(Tensor *input);
    float evaluate(vtensor &data_set, vector<vfloat> &target);
    void ClearGrad();
    bool save(const char *model_name);
    bool load(const char *model_name, int batch_size_ = 1);
    vector<Train_Args> getTrainArgs();
    void alloc_workspace();
private:
    string model;
    int layer_number;
    vector<LayerOption> opt_layer;
//    vector<Model_Layer> layer;
    Model_Layer *layer;
    vector<string> output_layer;
    unordered_map<string, Tensor*> terminal;
    vector<vector<int>> path;
    int batch_size;
    float *workspace;
    Forward_Args args;
    bool train;
};

class Trainer {
public:
    Trainer(Neural_Network *net, TrainerOption opt);
    vfloat train(Tensor &data, vfloat &target);
    vfloat train(vtensor &data_set, vector<vfloat> &target_set, int epoch);
    vfloat train_batch(Tensor &data, vfloat &target);
    vfloat train_batch(vtensor &data_set, vector<vfloat> &target_set, int epoch);
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
    int args_num;
};

#endif /* Neural_Network_hpp */
