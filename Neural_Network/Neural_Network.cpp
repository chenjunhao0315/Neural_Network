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
            //            printf("EuclideanLoss\n");
        } else if (type[0] == 'p' && type[1] == 'r') {
            opt["type"] = "PRelu";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
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
    for (int i = 0; i < output_layer.size(); ++i) {
        vector<int> route;
        string target_name = output_layer[i];
        for (int j = (int)opt_layer.size() - 1; j >= 0; --j) {
            if (target_name == opt_layer[j]["name"]) {
                route.push_back(j);
                target_name = opt_layer[j]["input_name"];
            }
        }
        path.push_back(route);
    }
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
        LayerType type = layer[i].getType();
        if (type == LayerType::Input) {
            fwrite("in", 2, 1, f);
        } else if (type == LayerType::Fullyconnected) {
            fwrite("fc", 2, 1, f);
        } else if (type == LayerType::Relu) {
            fwrite("re", 2, 1, f);
        } else if (type == LayerType::Softmax) {
            fwrite("sm", 2, 1, f);
        } else if (type == LayerType::Convolution) {
            fwrite("cn", 2, 1, f);
        } else if (type == LayerType::Pooling) {
            fwrite("po", 2, 1, f);
        } else if (type == LayerType::EuclideanLoss) {
            fwrite("el", 2, 1, f);
        } else if (type == LayerType::PRelu) {
            fwrite("pr", 2, 1, f);
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
    } else if (opt_["activation"] == "PRelu") {
        auto_opt["type"] = "PRelu";
        opt_layer.push_back(auto_opt);
    }
}

void Neural_Network::makeLayer() {
    unordered_map<string, int> id_table;
    printf("******Constructe Network******\n");
    for (int i = 0; i < opt_layer.size(); ++i) {
        printf("Create layer: %s\n", opt_layer[i]["type"].c_str());
        LayerOption &opt = opt_layer[i];
        if (opt.find("name") == opt.end())
            opt["name"] = to_string(i);
        id_table[opt["name"]] = i;
        if (i > 0) {
            if (opt.find("input_name") == opt.end())
                opt["input_name"] = (opt_layer[i - 1].find("name") == opt_layer[i - 1].end()) ? to_string(i - 1) : opt_layer[i - 1]["name"];
            opt["input_width"] = to_string(layer[id_table[opt["input_name"]]].getParameter(0));
            opt["input_height"] = to_string(layer[id_table[opt["input_name"]]].getParameter(1));
            opt["input_dimension"] = to_string(layer[id_table[opt["input_name"]]].getParameter(2));
        }
        string bias = opt["bias"].c_str();
        layer.push_back(Model_Layer(opt));
        layer_number++;
    }
    printf("*****************************\n");
    if (output_layer.empty())
        output_layer.push_back(opt_layer[opt_layer.size() - 1]["name"]);
    for (int i = 0; i < output_layer.size(); ++i) {
        vector<int> route;
        string target_name = output_layer[i];
        for (int j = (int)opt_layer.size() - 1; j >= 0; --j) {
            if (target_name == opt_layer[j]["name"]) {
                route.push_back(j);
                target_name = opt_layer[j]["input_name"];
            }
        }
        path.push_back(route);
    }
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
    size_t layer_size = layer.size();
    act = layer[0].Forward(input_tensor_);
    terminal[opt_layer[0]["name"]] = act;
    for (int i = 1; i < layer_size; ++i) {
//    for (int i = 0; i < layer.size(); ++i) {
//        if ((opt_layer[i].find("input_name") == opt_layer[i].end()) || opt_layer[i]["input_name"] == "default" || opt_layer[i]["input_name"] == "") {
//            act = layer[i].Forward(input_tensor_);
//        }
//        else {
            act = layer[i].Forward(terminal[opt_layer[i]["input_name"]]);
//        }
//        act->shape();
        terminal[opt_layer[i]["name"]] = act;
    }
    
    vfloat output = terminal[output_layer[0]]->toVector();
    size_t output_size = output.size();
    for (int i = 1; i < output_size; ++i) {
        vfloat temp = terminal[output_layer[i]]->toVector();
        size_t temp_size = temp.size();
        for (int j = 0; j < temp_size; ++j) {
            output.push_back(temp[j]);
        }
    }
    return output;
}

float Neural_Network::Backward(vfloat& target) {
    float loss = 0;
    if (model == "sequential") {
        int length = (int)layer.size();
        loss = layer[length - 1].Backward(target);
        for (int i = length - 2; i >= 0; --i) {
            layer[i].Backward();
        }
    } else if (model == "parallel") {
        if (target[0] == 1) {
            vfloat cls_pos{1};
            vfloat bbox_pos{target[1], target[2], target[3], target[4]};
            float loss_cls = layer[path[0][0]].Backward(cls_pos);
            float loss_bbox = layer[path[1][0]].Backward(bbox_pos);
            if (loss_cls > loss_bbox) {
                for (int i = 1; i < path[0].size(); ++i) {
                    layer[path[0][i]].Backward();
                }
            }
            else {
                for (int i = 1; i < path[1].size(); ++i) {
                    layer[path[1][i]].Backward();
                }
            }
            loss = loss_cls + loss_bbox * 0.5;
        }
        else if (target[0] == 0) {
            vfloat cls_neg{0};
            loss = layer[path[0][0]].Backward(cls_neg);
            for (int i = 1; i < path[0].size(); ++i) {
                layer[path[0][i]].Backward();
            }
        }
        else if (target[0] == -1) {
            vfloat bbox_part{target[1], target[2], target[3], target[4]};
            loss = layer[path[1][0]].Backward(bbox_part);
            for (int i = 1; i < path[1].size(); ++i) {
                layer[path[1][i]].Backward();
            }
            loss *= 0.5;
        }
        else if (target[0] == -2) {
            vfloat landmark;
            for (int i = 5; i <= 14; ++i) {
                landmark.push_back(target[i]);
            }
            loss = layer[path[2][0]].Backward(landmark);
            for (int i = 1; i < path[2].size(); ++i) {
                layer[path[2][i]].Backward();
            }
            loss *= 0.5;
        }
    }
    return loss;
}

vfloat Neural_Network::train(string method, float learning_rate, Tensor *input, vfloat &target) {
    float cost_lost = 0;
    
    Forward(input);
    cost_lost = Backward(target);
    
    int length = (int)layer.size();
    for (int i = 0; i < length; ++i) {
        layer[i].UpdateWeight(method, learning_rate);
    }
    
    return vfloat{cost_lost};
}

void Neural_Network::train(string method, float learning_rate, vtensor &data_set, vector<vfloat> &target_set, int epoch) {
    auto rng = std::default_random_engine {};
    vector<int> index;
    size_t data_set_size = data_set.size();
    for (int i = 0; i < data_set_size; ++i) {
        index.push_back(i);
    }
    for (int i = 0; i < epoch; ++i) {
        printf("Epoch %d Training[", i + 1);
        shuffle(index.begin(), index.end(), rng);
        for (int j = 0; j < data_set.size(); ++j) {
            //if (!(j % (data_set.size() / 20))) printf("*");
            //            Tensor input = data_set[a = index[j]];
            train(method, learning_rate, &(data_set[index[j]]), target_set[index[j]]);
        }
        printf("]\n");
        //float accuracy = evaluate(data_set, target_set);
        //printf("Accuracy: %.2f%%\n", accuracy * 100);
    }
}

vfloat Neural_Network::predict(Tensor *input) {
//    if (layer[layer.size() - 1].type_to_string() != "Softmax") {
//        printf("Last layer need to be softmax.\n");
//    }
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

vector<Tensor*> Neural_Network::getDetail() {
    vector<Tensor*> detail_set;
    
    for (int i = 0; i < layer.size(); ++i) {
        Tensor *kernel = nullptr, *biases = nullptr;
        kernel = layer[i].getKernel();
        biases = layer[i].getBiases();
        if (kernel)
            detail_set.push_back(kernel);
        if (biases)
            detail_set.push_back(biases);
    }
    return detail_set;
}

vector<vfloat> Neural_Network::getDetailParameter() {
    vector<vfloat> detail_parameter;
    
    for (int i = 0; i < layer.size(); ++i) {
        vfloat detail = layer[i].getDetailParameter();
        if (detail.size() == 7)
            detail_parameter.push_back(detail);
    }
    return detail_parameter;
}

void Neural_Network::UpdateNet() {
    for (int i = 0; i < layer.size(); ++i) {
        layer[i].Update();
        layer[i].ClearGrad();
    }
}

void Neural_Network::ClearGrad() {
    for (int i = 0; i < layer.size(); ++i) {
        layer[i].ClearGrad();
    }
}

Trainer::Trainer(Neural_Network *net, TrainerOption opt) {
    network = net;
    option = opt;
    learning_rate = (opt.find("learning_rate") == opt.end()) ? 0.001 : opt["learning_rate"];
    l1_decay = (opt.find("l1_decay") == opt.end()) ? 0.0 : opt["l1_decay"];
    l2_decay = (opt.find("l2_decay") == opt.end()) ? 0.0 : opt["l2_decay"];
    batch_size = (opt.find("batch_size") == opt.end()) ? 1 : (int)opt["batch_size"];
    method = (opt.find("method") == opt.end()) ? SGD : (Trainer::Method)opt["method"];
    momentum = (opt.find("momentum") == opt.end()) ? 0.9 : opt["momentum"];
    ro = (opt.find("ro") == opt.end()) ? 0.95 : opt["ro"];
    eps = (opt.find("eps") == opt.end()) ? 1e-6 : opt["eps"];
    beta_1 = (opt.find("beta_1") == opt.end()) ? 0.9 : opt["beta_1"];
    beta_2 = (opt.find("beta_2") == opt.end()) ? 0.999 : opt["beta_2"];
    iter = 0;
    gsum.resize(0);
    xsum.resize(0);
}

vfloat Trainer::train(vtensor &data_set, vector<vfloat> &target_set, int epoch) {
    auto rng = std::default_random_engine((unsigned)time(NULL));
    vector<int> index;
    float loss = 0;
    size_t data_set_size = data_set.size();
    for (int i = 0; i < data_set_size; ++i) {
        index.push_back(i);
    }
    for (int i = 0; i < epoch; ++i) {
        printf("Epoch %d Training[", i + 1);
        loss = 0;
        shuffle(index.begin(), index.end(), rng);
        for (int j = 0; j < data_set_size; ++j) {
            if (!(j % (data_set_size / 20))) printf("*");
            loss += train(data_set[index[j]], target_set[index[j]])[0];
            //            printf("%d ", index[j]);
        }
        printf("] ");
        printf("loss: %f\n", loss);
        //float accuracy = evaluate(data_set, target_set);
        //printf("Accuracy: %.2f%%\n", accuracy * 100);
    }
    return vfloat{loss};
}

vfloat Trainer::train(Tensor &data, vfloat &target) {
    network->Forward(&data);
    float loss = network->Backward(target);
    float l1_decay_loss = 0;
    float l2_decay_loss = 0;
    iter++;
    if ((iter % batch_size) == 0) {
        vector<Tensor*> detail_set = network->getDetail();
        vector<vfloat> detail_parameter = network->getDetailParameter();
        
        vector<float*> weight_list;
        vector<float*> grad_list;
        vector<int> len_list;
        vector<vfloat> decay_list;
        size_t detail_set_size = detail_set.size();
        for (int i = 0; i < detail_set_size; i += 2) {
            Tensor* kernel = detail_set[i];
            for (int j = 0; j < detail_parameter[i / 2][0]; ++j) {
                int len = detail_parameter[i / 2][1] * detail_parameter[i / 2][2] * detail_parameter[i / 2][3];
                weight_list.push_back(kernel[j].getWeight());
                grad_list.push_back(kernel[j].getDeltaWeight());
                len_list.push_back(len);
                decay_list.push_back(vfloat{detail_parameter[i / 2][4], detail_parameter[i / 2][5]});
            }
            Tensor* biases = detail_set[i + 1];
            weight_list.push_back(biases->getWeight());
            grad_list.push_back(biases->getDeltaWeight());
            len_list.push_back(detail_parameter[i / 2][4]);
            decay_list.push_back(vfloat{detail_parameter[i / 2][5], detail_parameter[i / 2][6]});
        }
        
        size_t len_list_size = len_list.size();
        if (gsum.empty() && (method != SGD || momentum > 0)) {
            for (int i = 0; i < len_list_size; ++i) {
                float *new_gsum = new float [len_list[i]];
                fill(new_gsum, new_gsum + len_list[i], 0);
                gsum.push_back(new_gsum);
                if (method == ADADELTA || method == ADAM) {
                    float *new_xsum = new float [len_list[i]];
                    fill(new_xsum, new_xsum + len_list[i], 0);
                    xsum.push_back(new_xsum);
                } else {
                    xsum.push_back(nullptr);
                }
            }
        }
        
        for (int i = 0; i < len_list_size; ++i) {
            float *weight = weight_list[i];
            float *grad = grad_list[i];
            
            float l1_decay_mul = decay_list[i][0];
            float l2_decay_mul = decay_list[i][1];
            float l1_decay_local = l1_decay * l1_decay_mul;
            float l2_decay_local = l2_decay * l2_decay_mul;
            
            int len = len_list[i];
            for (int j = 0; j < len; ++j) {
                l1_decay_loss += l1_decay_local * abs(weight[j]);
                l2_decay_loss += l2_decay_local * weight[j] * weight[j] / 2;
                float l1_grad = l1_decay_local * (weight[j] > 0 ? 1 : -1);
                float l2_grad = l2_decay_local * weight[j];
                float grad_ij = (l1_grad + l2_grad + grad[j]) / batch_size;
                
                float *gsumi = gsum[i];
                float *xsumi = xsum[i];
                if (method == ADADELTA) {
                    gsumi[j] = ro * gsumi[j] + (1 - ro) * grad_ij * grad_ij;
                    float delta_weight = -sqrt((xsumi[j] + eps) / (gsumi[j] + eps)) * grad_ij;
                    xsumi[j] = ro * xsumi[j] + (1 - ro) * delta_weight * delta_weight;
//                    grad[j] = delta_weight;
                    weight[j] += delta_weight;
                } else if (method == ADAM) {
//                    printf("gsumi: %.2f xsumi: %.2f\n", gsumi[j], xsumi[j]);
                    gsumi[j] = beta_1 * gsumi[j] + (1 - beta_1) * grad_ij;
                    xsumi[j] = beta_2 * xsumi[j] + (1 - beta_2) * grad_ij * grad_ij;
                    float nor_gsumi = gsumi[j] / (1 - pow(beta_1, (iter / batch_size) + 1));
                    float nor_xsumi = xsumi[j] / (1 - pow(beta_2, (iter / batch_size) + 1));
                    float delta_weight = -((nor_gsumi) / (sqrt(nor_xsumi) + eps)) * learning_rate;
//                    grad[j] = delta_weight;
                    weight[j] += delta_weight;
                } else { // SGD
                    if (momentum > 0) {
                        float delta_weight = momentum * gsumi[j] - learning_rate * grad_ij;
                        gsumi[j] = delta_weight;
                        weight[j] += delta_weight;
//                        grad[j] = delta_weight;
                    } else {
//                        grad[j] = grad_ij * learning_rate;
                        weight[j] += learning_rate * grad_ij;
                    }
                }
                grad[j] = 0;
            }
        }
        
    }
    return vfloat{loss};
}
