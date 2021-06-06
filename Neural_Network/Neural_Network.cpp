//
//  Neural_Network.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include "Neural_Network.hpp"

Neural_Network::~Neural_Network() {
    for (int i = 0; i < layer_number; ++i) {
    }
}

void Neural_Network::addLayer(LayerOption opt_) {
    LayerOption auto_opt;
    string type = opt_["type"];
    if (type == "Softmax") {
        auto_opt["type"] = "Fullyconnected";
        auto_opt["number_neurons"] = opt_["number_class"];
        opt_layer.push_back(auto_opt);
    }
    if (opt_["activation"] == "Relu") {
        opt_["bias"] = "0.1";
    }
    string bias = opt_["bias"].c_str();
    opt_layer.push_back(opt_);
    
    if (opt_["activation"] == "Relu") {
        auto_opt["type"] = "Relu";
        opt_layer.push_back(auto_opt);
    }
}

void Neural_Network::makeLayer() {
    printf("****Constructe Network****\n");
    for (int i = 0; i < opt_layer.size(); ++i) {
        printf("Create layer: %s\n", opt_layer[i]["type"].c_str());
        LayerOption opt = opt_layer[i];
        int a, b, c;
        if (i > 0) {
            opt["input_width"] = to_string(a = layer[i - 1].getParameter(0));
            opt["input_height"] = to_string(b = layer[i - 1].getParameter(1));
            opt["input_dimension"] = to_string(c = layer[i - 1].getParameter(2));
        }
        string bias = opt["bias"].c_str();
        layer.push_back(Model_Layer(opt));
        printf("end of for loop\n");
        ++layer_number;
    }
    printf("*************************\n");
}

void Neural_Network::shape() {
    printf("**********Shpae**********\n");
    for (int i = 0; i < layer.size(); ++i) {
        layer[i].shape();
    }
    printf("*************************\n");
}

Tensor* Neural_Network::Forward(Tensor *input_tensor_) {
    Tensor *act = layer[0].Forward(input_tensor_);
    for (int i = 0; i < layer.size(); ++i) {
        act = layer[i].Forward(act);
//        printf("Layer: %s\n", layer[i].getType().c_str());
//        act->showWeight();
    }
    return act;
}

float Neural_Network::Backward(float target) {
    int length = (int)layer.size();
    float loss = layer[length - 1].Backward(target);
    for (int i = length - 2; i >= 0; --i) {
        layer[i].Backward();
    }
    return loss;
}

vfloat Neural_Network::train(string method, float learning_rate, Tensor *input, float &target) {
    Forward(input);
    float cost_lost = Backward(target);
    
    int length = (int)layer.size();
    for (int i = 0; i < length; ++i) {
        layer[i].UpdateWeight(method, learning_rate);
    }
    return vfloat{cost_lost};
}

void Neural_Network::train(string method, float learning_rate, vtensor &data_set, vfloat &target, int epoch) {
    auto rng = std::default_random_engine {};
    vector<int> index;
    for (int i = 0; i < data_set.size(); ++i) {
        index.push_back(i);
    }
    for (int i = 0; i < epoch; ++i) {
        printf("Epoch %d Training[", i + 1);
        shuffle(index.begin(), index.end(), rng);
        for (int j = 0; j < data_set.size(); ++j) {
            if (!(j % (data_set.size() / 20))) printf("*");
//            Tensor input = data_set[a = index[j]];
            train(method, learning_rate, &(data_set[index[j]]), target[index[j]]);
        }
        printf("]\n");
        float accuracy = evaluate(data_set, target);
        printf("Accuracy: %.2f%%\n", accuracy * 100);
    }
}

vfloat Neural_Network::predict(Tensor *input) {
    if (layer[layer.size() - 1].getType() != "Softmax") {
        printf("Last layer need to be softmax.\n");
    }
    Tensor* output_tensor = Forward(input);
    float *output = output_tensor->getWeight();
    float max_value = output[0];
    int max_index = 0, index, length = (int)output_tensor->length();
    for (index = 1; index < length; ++index) {
        if (output[index] > max_value) {
            max_value = output[index];
            max_index = index;
        }
    }
    return vfloat{static_cast<float>(max_index), max_value};
}

float Neural_Network::evaluate(vtensor &data_set, vfloat &target) {
    int correct = 0;
    int data_number = (int)data_set.size();
    vfloat check;
    for (int i = 0; i < data_number; ++i) {
        check = predict(&data_set[i]);
        if ((int)check[0] == (int)target[i])
            correct++;
    }
    return (float)correct / data_number;
}
