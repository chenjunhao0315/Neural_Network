//
//  Neural_Network.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include "Neural_Network.hpp"

bool Neural_Network::load(const char *model_name) {
    FILE *f = fopen(model_name, "rb");
    if (!f)
        return false;
    int type_len;
    fread(&type_len, sizeof(int), 1, f);
    char *type = new char [type_len + 1];
    fread(type, sizeof(char), type_len, f);
    type[type_len] = '\0';
    model = string(type);
    
    fread(&layer_number, sizeof(int), 1, f);
    printf("Load %d layers\n", layer_number);
    for (int i = 0; i < layer_number; ++i) {
        int temp;
        float temp_f;
        LayerOption opt;
        char type[2] = {0};
        fread(type, 2, 1, f);
        fread(&temp, sizeof(int), 1, f);
        char *name_ = new char [temp + 1];
        fread(name_, sizeof(char), temp, f);
        name_[temp] = '\0';
        
        fread(&temp, sizeof(int), 1, f);
        char *input_name_ = new char [temp + 1];
        fread(input_name_, sizeof(char), temp, f);
        input_name_[temp] = '\0';
        opt["name"] = string(name_);
        opt["input_name"] = string(input_name_);
        if (type[0] == 'i' && type[1] == 'n') {
            opt["type"] = "Input";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            //            printf("Input\n");
        } else if (type[0] == 'f' && type[1] == 'c') {
            opt["type"] = "Fullyconnected";
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["number_neurons"] = to_string(temp);
            opt["input_width"] = "1";
            opt["input_height"] = "1";
            //            printf("Fullyconnected\n");
        } else if (type[0] == 'r' && type[1] == 'e') {
            opt["type"] = "Relu";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            //            printf("Relu\n");
        } else if (type[0] == 's' && type[1] == 'm') {
            opt["type"] = "Softmax";
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            opt["input_width"] = "1";
            //            printf("Softmax\n");
        } else if (type[0] == 'c' && type[1] == 'n') {
            opt["type"] = "Convolution";
            fread(&temp, sizeof(int), 1, f);
            opt["number_kernel"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["kernel_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["kernel_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["stride"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["padding"] = to_string(temp);
            //            printf("Convolution\n");
        } else if (type[0] == 'p' && type[1] == 'o') {
            opt["type"] = "Pooling";
            fread(&temp, sizeof(int), 1, f);
            opt["kernel_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["kernel_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["stride"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["padding"] = to_string(temp);
            //            printf("Pooling\n");
        } else if (type[0] == 'e' && type[1] == 'l') {
            opt["type"] = "EuclideanLoss";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            fread(&temp_f, sizeof(float), 1, f);
            opt["alpha"] = to_string(temp_f);
            //            printf("EuclideanLoss\n");
        }
        opt_layer.push_back(opt);
        layer.push_back(Model_Layer(opt));
        layer[i].load(f);
    }
    int output_number;
    fread(&output_number, sizeof(int), 1, f);
    for (int i = 0; i < output_number; ++i) {
        int len;
        fread(&len, sizeof(int), 1, f);
        char *output = new char [len + 1];
        fread(output, sizeof(char), len, f);
        output[len] = '\0';
        output_layer.push_back(string(output));
    }
    fclose(f);
    return true;
}

bool Neural_Network::save(const char *model_name) {
    FILE *f = fopen(model_name, "wb");
    if (!f)
        return false;
    int type_len = (int)strlen(model.c_str());
    fwrite(&type_len, sizeof(int), 1, f);
    char *type = new char [type_len];
    strcpy(type, model.c_str());
    fwrite(type, sizeof(char), type_len, f);
    
    printf("Save %d layers\n", layer_number);
    fwrite(&layer_number, sizeof(int), 1, f);
    for (int i = 0; i < layer_number; ++i) {
        string type = layer[i].getType();
        if (type == "Input") {
            fwrite("in", 2, 1, f);
        } else if (type == "Fullyconnected") {
            fwrite("fc", 2, 1, f);
        } else if (type == "Relu") {
            fwrite("re", 2, 1, f);
        } else if (type == "Softmax") {
            fwrite("sm", 2, 1, f);
        } else if (type == "Convolution") {
            fwrite("cn", 2, 1, f);
        } else if (type == "Pooling") {
            fwrite("po", 2, 1, f);
        } else if (type == "EuclideanLoss") {
            fwrite("el", 2, 1, f);
        }
        layer[i].save(f);
    }
    int output_number = (int)output_layer.size();
    fwrite(&output_number, sizeof(int), 1, f);
    for (int i = 0; i < output_number; ++i) {
        int len = (int)strlen(output_layer[i].c_str());
        fwrite(&len, sizeof(int), 1, f);
        char *output = new char [len];
        strcpy(output, output_layer[i].c_str());
        fwrite(output, sizeof(char), len, f);
    }
    fclose(f);
    return true;
}

void Neural_Network::addOutput(string name) {
    output_layer.push_back(name);
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
    printf("******Constructe Network******\n");
    for (int i = 0; i < opt_layer.size(); ++i) {
        printf("Create layer: %s\n", opt_layer[i]["type"].c_str());
        LayerOption &opt = opt_layer[i];
        if (opt.find("name") == opt.end())
            opt["name"] = to_string(i);
        if (i > 0) {
            if (opt.find("input_name") == opt.end())
                opt["input_name"] = (opt_layer[i - 1].find("name") == opt_layer[i - 1].end()) ? to_string(i - 1) : opt_layer[i - 1]["name"];
            opt["input_width"] = to_string(layer[i - 1].getParameter(0));
            opt["input_height"] = to_string(layer[i - 1].getParameter(1));
            opt["input_dimension"] = to_string(layer[i - 1].getParameter(2));
        }
        string bias = opt["bias"].c_str();
        layer.push_back(Model_Layer(opt));
        layer_number++;
    }
    printf("*****************************\n");
    if (output_layer.empty())
        output_layer.push_back(opt_layer[opt_layer.size() - 1]["name"]);
}

void Neural_Network::shape() {
    printf("Model type: \"%s\"\n", model.c_str());
    printf("------------------------------------------------------\n");
    printf("Layer(type)      Name       Input       Output Shape\n");
    printf("======================================================\n");
    for (int i = 0; i < layer.size(); ++i) {
        layer[i].shape();
    }
    printf("======================================================\n");
}

vfloat Neural_Network::Forward(Tensor *input_tensor_) {
    Tensor *act = input_tensor_;
    for (int i = 0; i < layer.size(); ++i) {
        if ((opt_layer[i].find("input_name") == opt_layer[i].end()) || opt_layer[i]["input_name"] == "default")
            act = layer[i].Forward(input_tensor_);
        else
            act = layer[i].Forward(terminal[opt_layer[i]["input_name"]]);
        
        terminal[opt_layer[i]["name"]] = act;
    }
    
    vfloat output = terminal[output_layer[0]]->toVector();
    for (int i = 1; i < output_layer.size(); ++i) {
        vfloat temp = terminal[output_layer[i]]->toVector();
        for (int j = 1; j < temp.size(); ++j) {
            output.push_back(temp[j]);
        }
    }
    return output;
}

float Neural_Network::Backward(vfloat& target) {
    int length = (int)layer.size();
    float loss = layer[length - 1].Backward(target);
    for (int i = length - 2; i >= 0; --i) {
        layer[i].Backward();
    }
    return loss;
}

vfloat Neural_Network::train(string method, float learning_rate, Tensor *input, vfloat &target) {
    float cost_lost = 0;
    
    if (model == "sequential") {
        Forward(input);
        cost_lost = Backward(target);
        
        int length = (int)layer.size();
        for (int i = 0; i < length; ++i) {
            layer[i].UpdateWeight(method, learning_rate);
        }
    } else {
        printf("testing...\n");
    }
    
    return vfloat{cost_lost};
}

void Neural_Network::train(string method, float learning_rate, vtensor &data_set, vector<vfloat> &target_set, int epoch) {
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
            train(method, learning_rate, &(data_set[index[j]]), target_set[index[j]]);
        }
        printf("]\n");
        float accuracy = evaluate(data_set, target_set);
        printf("Accuracy: %.2f%%\n", accuracy * 100);
    }
}

vfloat Neural_Network::predict(Tensor *input) {
    if (layer[layer.size() - 1].getType() != "Softmax") {
        printf("Last layer need to be softmax.\n");
    }
    vfloat output = Forward(input);
    float max_value = output[0];
    int max_index = 0, index, length = (int)output.size();
    for (index = 1; index < length; ++index) {
        if (output[index] > max_value) {
            max_value = output[index];
            max_index = index;
        }
    }
    return vfloat{static_cast<float>(max_index), max_value};
}

float Neural_Network::evaluate(vtensor &data_set, vector<vfloat> &target) {
    int correct = 0;
    int data_number = (int)data_set.size();
    vfloat check;
    for (int i = 0; i < data_number; ++i) {
        check = predict(&data_set[i]);
        if ((int)check[0] == (int)target[i][0])
            correct++;
    }
    return (float)correct / data_number;
}

Neural_Network::Neural_Network(string model_) {
    model = model_;
    layer_number = 0;
}
