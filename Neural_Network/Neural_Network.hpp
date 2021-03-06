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

struct network_structure {
    int width, height, dimension;
};

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
    Tensor** Forward(Tensor *input_tensor_, bool train = false);
    float Backward(Tensor *target);
    void extract(string name, Tensor &t);
    nn_status status();
    network_structure getStructure();
    void ClearGrad();
    Otter_Leader convert_to_otter();
    bool check_version(FILE *model);
    bool save_otter(const char *model_name, bool save_weight = false);
    bool save_dam(const char *model_name);
    bool save_ottermodel(const char *model_name);
    bool save_darknet(const char *weights_name, int cut_off = -1);
    bool load_otter(const char *model_structure, int batch_size = 1);
    bool load_dam(const char *model_weight);
    bool load_ottermodel(const char *model_name, int batch_size = 1);
    bool load_darknet(const char *weights_name);
    void shape();
    void show_detail();
    bool to_prototxt(const char *filename = "model.prototxt");
    void alloc_workspace();
    void constructGraph();
    int getBatchSize() {return batch_size;}
    int getOutputNum() {return max(1, (int)output_layer.size());}
    vector<Train_Args> getTrainArgs();
    vfloat predict(Tensor *input);
    float evaluate(vtensor &data_set, vtensor &target);
private:
    string model;
    BaseLayer **layer;
    Tensor **output_tensor;
    vector<string> output_layer;
    vector<LayerOption> opt_layer;
    unordered_map<string, vtensorptr> terminal;
    vector<vector<int>> path;
    int batch_size;
    int layer_number;
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

// Python interface
extern "C" {
    Tensor* create_tensor_init(int batch, int channel, int height, int width, float parameter);
    Tensor* create_tensor_array(float *data, int width, int height, int channel);
    void free_tensor(Tensor *t);
    Tensor* copy_tensor(Tensor *t);
    void tensor_show(Tensor *t);
    int tensor_batch(Tensor *t);
    int tensor_channel(Tensor *t);
    int tensor_height(Tensor *t);
    int tensor_width(Tensor *t);
    float tensor_get(Tensor *t, int key);
    void tensor_set(Tensor *t, int key, float value);
    float* tensor_get_weight(Tensor *t);

    Neural_Network* create_network(const char *model_name);
    void free_network(Neural_Network *net);
    void network_load_ottermodel(Neural_Network *net, const char *ottermodel);
    Tensor** network_forward(Neural_Network *net, Tensor *data);
    int network_getoutputnum(Neural_Network *net);
    void network_shape(Neural_Network *net);
}
#endif /* Neural_Network_hpp */
