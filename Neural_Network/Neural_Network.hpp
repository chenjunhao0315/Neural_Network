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
#include <sstream>

#include "Layer.hpp"
#include "Otter.hpp"

typedef map<string, float> TrainerOption;

class Neural_Network {
public:
    enum nn_status {
        OK,
        ERROR
    };
    ~Neural_Network();
    Neural_Network(string model_ = "sequential");
    void addLayer(LayerOption opt_);
    void addOutput(string name);
    void compile(int batch_size_ = 1);
    void shape();
    void show_detail();
    nn_status status();
    vtensorptr Forward(Tensor *input_tensor_, bool train = false);
    float Backward(Tensor *target);
    void ClearGrad();
    Otter_Leader convert_to_otter();
    bool check_version(FILE *model);
    bool save(const char *model_name);
    bool save_darknet(const char *weights_name, int cut_off = -1);
    bool save_otter(const char *model_name, bool save_weight = false);
    bool save_dam(const char *model_name);
    bool save_ottermodel(const char *model_name);
    bool load(const char *model_name, int batch_size_ = 1);
    bool load_darknet(const char *weights_name);
    bool load_otter(const char *model_structure, const char *model_weight = nullptr);
    bool load_dam(const char *model_weight);
    bool load_ottermodel(const char *model_name);
    bool to_prototxt(const char *filename = "model.prototxt");
    void alloc_workspace();
    void constructGraph();
    int getBatchSize() {return batch_size;}
    vector<Train_Args> getTrainArgs();
    vfloat predict(Tensor *input);
    float evaluate(vtensor &data_set, vtensor &target);
private:
    string model;
    int layer_number;
    vector<LayerOption> opt_layer;
    BaseLayer **layer;
    vtensorptr output;
    vector<string> output_layer;
    unordered_map<string, Tensor*> terminal;
    vector<vector<int>> path;
    int batch_size;
    float *workspace;
    int version_major = 3;
    int version_minor = 0;
};

class Trainer {
public:
    Trainer(Neural_Network *net, TrainerOption opt);
    vfloat train(Tensor &data, Tensor &target);
    vfloat train(vtensor &data_set, vtensor &target_set, int epoch);
    vfloat train_batch(Tensor &data, Tensor &target);
    vfloat train_batch(vtensor &data_set, vtensor &target_set, int epoch);
    void decade(float rate);
    enum Method {
        SGD,
        ADADELTA,
        ADAM
    };
    
    enum Policy {
        CONSTANT,
        STEP,
        STEPS,
        EXP,
        POLY,
        RANDOM,
        SIG
    };
private:
    float get_learning_rate();
    
    Neural_Network *network;
    TrainerOption option;
    float learning_rate;
    float l1_decay;
    float l2_decay;
    int seen;
    int batch_num;
    int batch_size;
    int sub_division;
    int max_batches;
    Method method;
    Policy policy;
    float momentum;
    
    int warmup;
    float power;
    
    float ro;
    
    float eps;
    float beta_1;
    float beta_2;
    
    int step;
    float scale;
    
    int steps_num;
    Tensor steps;
    Tensor scales;
    
    float gamma;
    
    vector<float*> gsum;
    vector<float*> xsum;
    int args_num;
};

#endif /* Neural_Network_hpp */
