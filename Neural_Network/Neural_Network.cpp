//
//  Neural_Network.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include "Neural_Network.hpp"

bool Neural_Network::load(const char *model_) {
    FILE *model = fopen(model_, "rb");
    if (!model)
        return false;
    fread(&layer_number, sizeof(int), 1, model);
    printf("Layer number: %d\n", layer_number);
    for (int i = 0; i < layer_number; ++i) {
        int temp;
        LayerOption opt;
        char type[2] = {0};
        fread(type, 2, 1, model);
        if (type[0] == 'i' && type[1] == 'n') {
            opt["type"] = "Input";
            fread(&temp, sizeof(int), 1, model);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["input_dimension"] = to_string(temp);
            printf("Input\n");
        } else if (type[0] == 'f' && type[1] == 'c') {
            opt["type"] = "Fullyconnected";
            fread(&temp, sizeof(int), 1, model);
            opt["input_dimension"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["number_neurons"] = to_string(temp);
            opt["input_width"] = "1";
            opt["input_height"] = "1";
            printf("Fullyconnected\n");
        } else if (type[0] == 'r' && type[1] == 'e') {
            opt["type"] = "Relu";
            fread(&temp, sizeof(int), 1, model);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["input_dimension"] = to_string(temp);
            printf("Relu\n");
        } else if (type[0] == 's' && type[1] == 'm') {
            opt["type"] = "Softmax";
            fread(&temp, sizeof(int), 1, model);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["input_dimension"] = to_string(temp);
            opt["input_width"] = "1";
            printf("Softmax\n");
        } else if (type[0] == 'c' && type[1] == 'n') {
            opt["type"] = "Convolution";
            fread(&temp, sizeof(int), 1, model);
            opt["number_kernel"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["kernel_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["input_dimension"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["kernel_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["stride"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["padding"] = to_string(temp);
            printf("Convolution\n");
        } else if (type[0] == 'p' && type[1] == 'o') {
            opt["type"] = "Pooling";
            fread(&temp, sizeof(int), 1, model);
            opt["kernel_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["input_dimension"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["kernel_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["stride"] = to_string(temp);
            fread(&temp, sizeof(int), 1, model);
            opt["padding"] = to_string(temp);
            printf("Pooling\n");
        }
        layer.push_back(Model_Layer(opt));
        layer[i].load(model);
    }
    fclose(model);
    return true;
}

bool Neural_Network::save() {
    FILE *model = fopen("model.bin", "wb");
    if (!model)
        return false;
    printf("write Layer number: %d\n", layer_number);
    fwrite(&layer_number, sizeof(int), 1, model);
    for (int i = 0; i < layer_number; ++i) {
        string type = layer[i].getType();
        if (type == "Input") {
            fwrite("in", 2, 1, model);
        } else if (type == "Fullyconnected") {
            fwrite("fc", 2, 1, model);
        } else if (type == "Relu") {
            fwrite("re", 2, 1, model);
        } else if (type == "Softmax") {
            fwrite("sm", 2, 1, model);
        } else if (type == "Convolution") {
            fwrite("cn", 2, 1, model);
        } else if (type == "Pooling") {
            fwrite("po", 2, 1, model);
        }
        layer[i].save(model);
    }
    
    fclose(model);
    return true;
}

void Neural_Network::addLayer(LayerOption opt_) {
    LayerOption auto_opt;
    string type = opt_["type"];
    if (opt_["activation"] == "Relu") {
        opt_["bias"] = "0.1";
    }
    string bias = opt_["bias"].c_str();
    opt_layer.push_back(opt_);
    
    if (opt_["activation"] == "Relu") {
        auto_opt["type"] = "Relu";
        opt_layer.push_back(auto_opt);
    } else if (opt_["activation"] == "Softmax") {
        auto_opt["type"] = "Softmax";
        auto_opt["number_class"] = auto_opt["number_neurons"];
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
        layer_number++;
    }
    printf("*************************\n");
}

void Neural_Network::shape() {
    printf("Layer(type)         Output Shape\n");
    printf("======================================\n");
    for (int i = 0; i < layer.size(); ++i) {
        layer[i].shape();
    }
    printf("======================================\n");
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
