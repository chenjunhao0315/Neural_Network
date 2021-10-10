//
//  Neural_Network.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include "Neural_Network.hpp"

Neural_Network::~Neural_Network() {
    for (int i = layer_number; i--; )
        delete layer[i];
    delete [] layer;
    delete [] workspace;
}

bool Neural_Network::load(const char *model_name, int batch_size_) {
    batch_size = batch_size_;
    FILE *f = fopen(model_name, "rb");
    if (!f)
        return false;
    
    int check_major, check_minor;
    fread(&check_major, sizeof(int), 1, f);
    fread(&check_minor, sizeof(int), 1, f);
    if (check_major != version_major) {
        fprintf(stderr, "Neural Network: v%d.%d\n", version_major, version_minor);
        fprintf(stderr, "Model: v%d.%d\n", check_major, check_minor);
        fprintf(stderr, "There are significant change in weight arrangement, model can not be used!\n");
        if (check_major == 3 && version_major == 4)
            fprintf(stderr, "Please use the model converter to convert the model from v3 to v4!\n");
        exit(-100);
        return false;
    } else if (check_major == version_major && check_minor > version_minor) {
        fprintf(stderr, "Neural Network: v%d.%d\n", version_major, version_minor);
        fprintf(stderr, "Model: v%d.%d\n", check_major, check_minor);
        fprintf(stderr, "There are maybe existed unsupport layer!\n");
    }
    
    int type_len;
    fread(&type_len, sizeof(int), 1, f);
    char *type = new char [type_len + 1];
    fread(type, sizeof(char), type_len, f);
    type[type_len] = '\0';
    model = string(type);
    
    fread(&layer_number, sizeof(int), 1, f);
    printf("Load %d layers\n", layer_number);
    
    layer = new BaseLayer* [layer_number];
    
    for (int i = 0; i < layer_number; ++i) {
        int temp;
        LayerOption opt;
        opt["batch_size"] = to_string(batch_size);
        char type[2] = {0};
        fread(type, 2, 1, f);
        fread(&temp, sizeof(int), 1, f);
        char name_[temp + 1];
        fread(name_, sizeof(char), temp, f);
        name_[temp] = '\0';
        
        fread(&temp, sizeof(int), 1, f);
        char input_name_[temp + 1];
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
        } else if (type[0] == 'f' && type[1] == 'c') {
            opt["type"] = "FullyConnected";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["number_neurons"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            if (temp)
                opt["batchnorm"] = "true";
        } else if (type[0] == 'r' && type[1] == 'e') {
            opt["type"] = "Relu";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
        } else if (type[0] == 's' && type[1] == 'm') {
            opt["type"] = "Softmax";
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            opt["input_width"] = "1";
        } else if (type[0] == 'c' && type[1] == 'n') {
            opt["type"] = "Convolution";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["number_kernel"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["kernel_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["kernel_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["stride"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["padding"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            if (temp)
                opt["batchnorm"] = "true";
        } else if (type[0] == 'p' && type[1] == 'o') {
            opt["type"] = "Pooling";
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["kernel_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["kernel_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["stride"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["padding"] = to_string(temp);
        } else if (type[0] == 'e' && type[1] == 'l') {
            opt["type"] = "EuclideanLoss";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
        } else if (type[0] == 'p' && type[1] == 'r') {
            opt["type"] = "PRelu";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
        } else if (type[0] == 's' && type[1] == 'c') {
            opt["type"] = "ShortCut";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["shortcut_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["shortcut_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["shortcut_dimension"] = to_string(temp);
            int len;
            fread(&len, sizeof(int), 1, f);
            char *sc = new char [len + 1];
            fread(sc, sizeof(char), len, f);
            sc[len] = '\0';
            opt["shortcut"] = string(sc);
        } else if (type[0] == 'l' && type[1] == 'r') {
            opt["type"] = "LRelu";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
        } else if (type[0] == 's' && type[1] == 'i') {
            opt["type"] = "Sigmoid";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
        } else if (type[0] == 'b' && type[1] == 'n') {
            opt["type"] = "BatchNormalization";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
        }else if (type[0] == 'u' && type[1] == 'p') {
            opt["type"] = "UpSample";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            int stride = temp;
            fread(&temp, sizeof(int), 1, f);
            if (temp)
                stride = -stride;
            opt["stride"] = to_string(stride);
        } else if (type[0] == 'c' && type[1] == 'c') {
            opt["type"] = "Concat";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["concat_num"] = to_string(temp);
            for (int i = 1; i <= temp; ++i) {
                int concat_parameter;
                fread(&concat_parameter, sizeof(int), 1, f);
                opt["concat_" + to_string(i) + "_width"] = to_string(concat_parameter);
                fread(&concat_parameter, sizeof(int), 1, f);
                opt["concat_" + to_string(i) + "_height"] = to_string(concat_parameter);
                fread(&concat_parameter, sizeof(int), 1, f);
                opt["concat_" + to_string(i) + "_dimension"] = to_string(concat_parameter);
            }
            fread(&temp, sizeof(int), 1, f);
            opt["splits"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["split_id"] = to_string(temp);
            int len;
            fread(&len, sizeof(int), 1, f);
            if (len) {
                char *cc = new char [len + 1];
                fread(cc, sizeof(char), len, f);
                cc[len] = '\0';
                opt["concat"] = string(cc);
                stringstream concat_list(opt["concat"]);
                for (int i = 1; i <= atoi(opt["concat_num"].c_str()); ++i) {
                    string concat_name;
                    getline(concat_list, concat_name, ',');
                    concat_name.erase(std::remove_if(concat_name.begin(), concat_name.end(), [](unsigned char x) { return std::isspace(x); }), concat_name.end());
                    opt["concat_" + to_string(i) + "_name"] = concat_name;
                }
            }
        } else if (type[0] == 'y' && type[1] == '3') {
            opt["type"] = "YOLOv3";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["total_anchor_num"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["anchor_num"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["classes"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["max_boxes"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["net_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["net_height"] = to_string(temp);
        } else if (type[0] == 'y' && type[1] == '4') {
            opt["type"] = "YOLOv4";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["total_anchor_num"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["anchor_num"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["classes"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["max_boxes"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["net_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["net_height"] = to_string(temp);
            float scale_x_y;
            fread(&scale_x_y, sizeof(float), 1, f);
            opt["scale_x_y"] = to_string(scale_x_y);
        } else if (type[0] == 'm' && type[1] == 'i') {
            opt["type"] = "Mish";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
        } else if (type[0] == 's' && type[1] == 'w') {
            opt["type"] = "Swish";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
        } else if (type[0] == 'd' && type[1] == 'o') {
            opt["type"] = "Dropout";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
            float prob;
            fread(&prob, sizeof(float), 1, f);
            opt["probability"] = to_string(temp);
        } else if (type[0] == 'a' && type[1] == 'p') {
            opt["type"] = "AvgPooling";
            fread(&temp, sizeof(int), 1, f);
            opt["input_width"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_height"] = to_string(temp);
            fread(&temp, sizeof(int), 1, f);
            opt["input_dimension"] = to_string(temp);
        } else {
            fprintf(stderr, "[Neural_Network] Load unknown layer!\n");
        }
        opt_layer.push_back(opt);
        layer[i] = LayerRegistry::CreateLayer(opt);
        layer[i]->load(f);
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
    constructGraph();
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
    return status();
}

bool Neural_Network::save(const char *model_name) {
    FILE *f = fopen(model_name, "wb");
    if (!f)
        return false;
    
    fwrite(&version_major, sizeof(int), 1, f);
    fwrite(&version_minor, sizeof(int), 1, f);
    
    int type_len = (int)strlen(model.c_str());
    fwrite(&type_len, sizeof(int), 1, f);
    char *type = new char [type_len];
    strcpy(type, model.c_str());
    fwrite(type, sizeof(char), type_len, f);
    
    printf("Save %d layers\n", layer_number);
    fwrite(&layer_number, sizeof(int), 1, f);
    for (int i = 0; i < layer_number; ++i) {
        LayerType type = layer[i]->type;
        if (type == LayerType::Input) {
            fwrite("in", 2, 1, f);
        } else if (type == LayerType::FullyConnected) {
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
        } else if (type == LayerType::ShortCut) {
            fwrite("sc", 2, 1, f);
        } else if (type == LayerType::LRelu) {
            fwrite("lr", 2, 1, f);
        } else if (type == LayerType::Sigmoid) {
            fwrite("si", 2, 1, f);
        } else if (type == LayerType::BatchNormalization) {
            fwrite("bn", 2, 1, f);
        } else if (type == LayerType::UpSample) {
            fwrite("up", 2, 1, f);
        } else if (type == LayerType::Concat) {
            fwrite("cc", 2, 1, f);
        } else if (type == LayerType::Yolov3) {
            fwrite("y3", 2, 1, f);
        } else if (type == LayerType::Yolov4) {
            fwrite("y4", 2, 1, f);
        } else if (type == LayerType::Mish) {
            fwrite("mi", 2, 1, f);
        } else if (type == LayerType::Swish) {
            fwrite("sw", 2, 1, f);
        } else if (type == LayerType::Dropout) {
            fwrite("do", 2, 1, f);
        } else if (type == LayerType::AvgPooling) {
            fwrite("ap", 2, 1, f);
        }
        layer[i]->save(f);
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

bool Neural_Network::save_darknet(const char *weights_name, int cut_off) {
    FILE *f = fopen(weights_name, "wb");
    if (!f)
        return false;
    int major = 0;
    int minor = 2;
    int revision = 0;
    int seen = 32013312;
    fwrite(&major, sizeof(int), 1, f);
    fwrite(&minor, sizeof(int), 1, f);
    fwrite(&revision, sizeof(int), 1, f);
    if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000){
        fwrite(&seen, sizeof(size_t), 1, f);
    } else {
        fwrite(&seen, sizeof(int), 1, f);
    }
    
    int cut_off_num = (cut_off == -1) ? layer_number : cut_off;
    
    for (int i = 0; i < cut_off_num; ++i) {
        LayerOption &opt = opt_layer[i];
        if (opt["type"] == "Convolution") {
            if (opt.find("batchnorm") != opt.end()) {
                layer[i + 1]->save_raw(f);
            }
            layer[i]->save_raw(f);
        }
    }
    
    return true;
}

bool Neural_Network::load_darknet(const char *weights_name) {
    FILE *f = fopen(weights_name, "rb");
    if (!f)
        return false;
    fseek(f, 0, SEEK_END);
    size_t check = ftell(f);
    fseek(f, 0, SEEK_SET);
    int major;
    int minor;
    int revision;
    int seen;
    fread(&major, sizeof(int), 1, f);
    fread(&minor, sizeof(int), 1, f);
    fread(&revision, sizeof(int), 1, f);
    if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000){
        fread(&seen, sizeof(size_t), 1, f);
    } else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, f);
        seen = iseen;
    }
    printf("Major: %d Minor: %d Revision: %d Seen: %d\n", major, minor, revision, seen);
    
    for (int i = 0; i < opt_layer.size(); ++i) {
        LayerOption &opt = opt_layer[i];
        size_t test = ftell(f);
        cout << opt["name"] << ": " << check << " / " << test;
        if (opt["type"] == "Convolution") {
            if (opt.find("batchnorm") != opt.end()) {
                layer[i + 1]->load_raw(f);
            }
            layer[i]->load_raw(f);
        }
        test = ftell(f);
        cout << " -> " << test << endl;
    }
    size_t end = ftell(f);
    printf("End: %zu %zu\n", check, end);
    return end == check;
}

bool Neural_Network::load_otter(const char *model_structure, const char *model_weight) {
    Otter_Leader leader;
    leader.read_project(model_structure);
    
    model = leader.getName();
    for (int i = 0; i < leader.members(); ++i) {
        vector<Stick> materials = leader.getMaterial(i);
        string type = leader.getTeamName(i);
        LayerOption opt;
        opt["type"] = type;
        for (int j = 0; j < materials.size(); ++j) {
            Stick material = materials[j];
            opt[material.type] = material.info;
        }
        this->addLayer(opt);
    }
    vector<Option> option = leader.getOption();
    for (int i = 0; i < option.size(); ++i) {
        if (option[i].type == "output")
            this->addOutput(option[i].info);
    }
    this->compile();
    if (model_weight)
        this->load_dam(model_weight);
    return true;
}

bool Neural_Network::load_dam(const char *model_weight) {
    FILE *weights = fopen(model_weight, "r");
    
    int check_major, check_minor;
    fread(&check_major, sizeof(int), 1, weights);
    fread(&check_minor, sizeof(int), 1, weights);
    if (check_major != version_major) {
        printf("Neural Network: v%d.%d\n", version_major, version_minor);
        printf("Model: v%d.%d\n", check_major, check_minor);
        printf("There are significant change in weight arrangement, model can not be used!\n");
        if (check_major == 3 && version_major == 4)
            fprintf(stderr, "Please use the model converter to convert the model from v3 to v4!\n");
        exit(-100);
        return false;
    } else if (check_major == version_major && check_minor > version_minor) {
        printf("Neural Network: v%d.%d\n", version_major, version_minor);
        printf("Model: v%d.%d\n", check_major, check_minor);
        printf("There are maybe existed unsupport layer!\n");
    }
    
    for (int i = 0; i < layer_number; ++i) {
        layer[i]->load_raw(weights);
    }
    
    fclose(weights);
    return true;
}

bool Neural_Network::save_otter(const char *model_name) {
    Otter_Leader leader(model);
    for (int i = 0; i < output_layer.size(); ++i) {
        leader.addOption(Option("output", output_layer[i]));
    }
    
    for (int i = 0; i < layer_number; ++i) {
        LayerOption &opt = opt_layer[i];
        Otter team(opt["type"]), param("Param");
        team.addMaterial(Stick("name", opt["name"]));
        if (i) team.addMaterial(Stick("input_name", opt["input_name"]));
        if (opt.find("batchnorm") != opt.end()) {
            team.addMaterial(Stick("batchnorm", opt["batchnorm"]));
            ++i;
        }
        if (opt.find("activation") != opt.end()) {
            team.addMaterial(Stick("activation", opt["activation"]));
            ++i;
        }
        // param
        Parameter layer_param = LayerParameter::getParameter(opt["type"]);
        vector<ParameterData> data = layer_param.getParameter();
        for (int j = 0; j < data.size(); ++j) {
            ParameterData &par = data[j];
            if (opt.find(par.name) != opt.end())
                param.addMaterial(Stick(par.name, opt[par.name]));
        }
        if (!param.idle())
            team.addPartner(param);
        leader.addTeam(team);
    }
    leader.save_project(model_name);
    
    string weights(model_name);
    size_t pos = weights.find('.');
    weights = weights.substr(0, pos);
    weights.append(".dam");
    this->save_dam(weights.c_str());
    return true;
}

bool Neural_Network::save_dam(const char *model_name) {
    FILE *weights = fopen(model_name, "wb");
    fwrite(&version_major, sizeof(int), 1, weights);
    fwrite(&version_minor, sizeof(int), 1, weights);
    
    for (int i = 0; i < layer_number; ++i) {
        layer[i]->save_raw(weights);
    }
    
    fclose(weights);
    return true;
}

void Neural_Network::addOutput(string name) {
    output_layer.push_back(name);
}

void Neural_Network::addLayer(LayerOption opt_) {
    LayerOption auto_opt;
    string type = opt_["type"];
    if (opt_.find("activation") != opt_.end()) {
        if (opt_["activation"] == "Relu") {
            opt_["bias"] = "0.1";
        }
    }
    string bias = opt_["bias"].c_str();
    
    opt_layer.push_back(opt_);
    
    if (opt_.find("batchnorm") != opt_.end()) {
        auto_opt["type"] = "BatchNormalization";
        if (opt_.find("name") != opt_.end()) {
            auto_opt["name"] = "bn_" + opt_["name"];
        }
        opt_layer.push_back(auto_opt);
    }
    
    auto_opt.clear();
    if (opt_.find("activation") != opt_.end()) {
        string method = opt_["activation"];
        if (method == "Relu") {
            if (opt_.find("name") != opt_.end()) {
                auto_opt["name"] = "re_" + opt_["name"];
            }
            auto_opt["type"] = "Relu";
            opt_layer.push_back(auto_opt);
        } else if (method == "Softmax") {
            if (opt_.find("name") != opt_.end()) {
                auto_opt["name"] = "sm_" + opt_["name"];
            }
            auto_opt["type"] = "Softmax";
            auto_opt["number_class"] = auto_opt["number_neurons"];
            opt_layer.push_back(auto_opt);
        } else if (method == "PRelu") {
            if (opt_.find("name") != opt_.end()) {
                auto_opt["name"] = "pr_" + opt_["name"];
            }
            auto_opt["type"] = "PRelu";
            opt_layer.push_back(auto_opt);
        } else if (method == "LRelu") {
            if (opt_.find("name") != opt_.end()) {
                auto_opt["name"] = "lr_" + opt_["name"];
            }
            auto_opt["type"] = "LRelu";
            opt_layer.push_back(auto_opt);
        } else if (method == "Sigmoid") {
            if (opt_.find("name") != opt_.end()) {
                auto_opt["name"] = "sg_" + opt_["name"];
            }
            auto_opt["type"] = "Sigmoid";
            opt_layer.push_back(auto_opt);
        } else if (method == "Mish") {
            if (opt_.find("name") != opt_.end()) {
                auto_opt["name"] = "mi_" + opt_["name"];
            }
            auto_opt["type"] = "Mish";
            opt_layer.push_back(auto_opt);
        } else if (method == "Swish") {
            if (opt_.find("name") != opt_.end()) {
                auto_opt["name"] = "sw_" + opt_["name"];
            }
            auto_opt["type"] = "Swish";
            opt_layer.push_back(auto_opt);
        }
    }
}

void Neural_Network::compile(int batch_size_) {
    unordered_map<string, int> id_table;
    batch_size = batch_size_;
    layer = new BaseLayer* [opt_layer.size()];
    printf("******Constructe Network******\n");
    for (int i = 0; i < opt_layer.size(); ++i) {
        printf("Create layer: %s\n", opt_layer[i]["type"].c_str());
        LayerOption &opt = opt_layer[i];
        opt["batch_size"] = to_string(batch_size);
        if (opt.find("name") == opt.end())
            opt["name"] = to_string(i);
        id_table[opt["name"]] = i;
        if (i > 0) {
            if (opt.find("input_name") == opt.end())
                opt["input_name"] = (opt_layer[i - 1].find("name") == opt_layer[i - 1].end()) ? to_string(i - 1) : opt_layer[i - 1]["name"];
            opt["input_width"] = to_string(layer[id_table[opt["input_name"]]]->getParameter(0));
            opt["input_height"] = to_string(layer[id_table[opt["input_name"]]]->getParameter(1));
            opt["input_dimension"] = to_string(layer[id_table[opt["input_name"]]]->getParameter(2));
        }
        string bias = opt["bias"].c_str();
        
        if (opt["type"] == "Concat") {
            int concat_num = 0;
            if (opt.find("concat") != opt.end()) {
                concat_num = (int)count(opt["concat"].begin(), opt["concat"].end(), ',') + 1;
                stringstream concat_list(opt["concat"]);
                
                for (int i = 1; i <= concat_num; ++i) {
                    string concat_name;
                    getline(concat_list, concat_name, ',');
                    concat_name.erase(std::remove_if(concat_name.begin(), concat_name.end(), [](unsigned char x) { return std::isspace(x); }), concat_name.end());
                    opt["concat_" + to_string(i) + "_name"] = concat_name;
                    opt["concat_" + to_string(i) + "_width"] = to_string(layer[id_table[concat_name]]->getParameter(0));
                    opt["concat_" + to_string(i) + "_height"] = to_string(layer[id_table[concat_name]]->getParameter(1));
                    opt["concat_" + to_string(i) + "_dimension"] = to_string(layer[id_table[concat_name]]->getParameter(2));
                }
            }
            opt["concat_num"] = to_string(concat_num);
        } else if (opt["type"] == "YOLOv3" || opt["type"] == "YOLOv4") {
            opt["net_width"] = to_string(layer[0]->getParameter(0));
            opt["net_height"] = to_string(layer[0]->getParameter(1));
        } else if (opt["type"] == "ShortCut") {
            opt["shortcut_width"] = to_string(layer[id_table[opt["shortcut"]]]->getParameter(0));
            opt["shortcut_height"] = to_string(layer[id_table[opt["shortcut"]]]->getParameter(1));
            opt["shortcut_dimension"] = to_string(layer[id_table[opt["shortcut"]]]->getParameter(2));
        }
        layer[i] = LayerRegistry::CreateLayer(opt);
        layer_number++;
    }
    printf("*****************************\n");
//    if (output_layer.empty())
//        output_layer.push_back(opt_layer[opt_layer.size() - 1]["name"]);
    constructGraph();
    
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
    printf("-------------------------------------------------------------\n");
    printf("Layer(type)      Name          Input          Output Shape\n");
    printf("=============================================================\n");
    for (int i = 0; i < layer_number; ++i) {
        layer[i]->shape();
    }
    printf("=============================================================\n");
}

void Neural_Network::show_detail() {
    for (int i = 0; i < layer_number; ++i) {
        layer[i]->show_detail();
    }
}

Neural_Network::nn_status Neural_Network::status() {
    return (layer_number > 0) ? ((layer[0]->type == LayerType::Input) ? nn_status::OK : nn_status::ERROR) : nn_status::ERROR;
}

bool Neural_Network::to_prototxt(const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f)
        return false;
    // Model name
    fprintf(f, "name: \"%s\"\n", model.c_str());
    
    unordered_map<string, int> id_table;
    for (int i = 0; i < layer_number; ++i) {
        id_table[opt_layer[i]["name"]] = i;
    }
    vector<LayerOption> refine_struct(layer_number);
    for (int i = 0; i < layer_number; ) {
        int count = 1;
        LayerOption &refine = refine_struct[i];
        LayerOption &opt = opt_layer[i];
        refine["name"] = (i == 0) ? "data" : opt["name"];
        if (refine_struct[id_table[opt["input_name"]]].find("name") == refine_struct[id_table[opt["input_name"]]].end()) {
            cout << opt["input_name"] << endl;
            cout << "input_id: " << id_table[opt["input_name"]] << endl;
        }
        refine["input_name"] = refine_struct[id_table[opt["input_name"]]]["name"];
        if (opt.find("batchnorm") != opt.end()) {
            refine_struct[i + 1]["name"] = opt["name"];
            refine_struct[i + 1]["input_name"] = opt["name"];
            ++count;
        }
        if (opt.find("activation") != opt.end()) {
            refine_struct[i + count]["name"] = opt["name"];
            refine_struct[i + count]["input_name"] = opt["name"];
            ++count;
        }
        i += count;
    }
    
    for (int i = 0; i < layer_number; ++i) {
        layer[i]->to_prototxt(f, i, refine_struct, id_table);
    }
    
    fclose(f);
    return true;
}

void Neural_Network::constructGraph() {
    alloc_workspace();
    for (int i = 0; i < layer_number; ++i) {
        LayerOption &opt = opt_layer[i];
        vtensorptr extra_tensor;
        if (opt["type"] == "ShortCut") {
            extra_tensor.push_back(terminal[opt["shortcut"]]);
        } else if (opt["type"] == "Concat") {
            extra_tensor.push_back(terminal[opt["input_name"]]);
            int concat_num = 0;
            if (opt.find("concat") != opt.end()) {
                concat_num = (int)count(opt["concat"].begin(), opt["concat"].end(), ',') + 1;
                stringstream concat_list(opt["concat"]);
                for (int i = 1; i <= concat_num; ++i) {
                    string concat_name;
                    getline(concat_list, concat_name, ',');
                    concat_name.erase(std::remove_if(concat_name.begin(), concat_name.end(), [](unsigned char x) { return std::isspace(x); }), concat_name.end());
                    extra_tensor.push_back(terminal[concat_name]);
                }
            }
        }
        Tensor *act = layer[i]->connectGraph(terminal[opt["input_name"]], extra_tensor, workspace);
        terminal[opt["name"]] = act;
    }
    
    int output_size = (int)output_layer.size();
    if (output_size) {
        output.reserve(output_size);
        output.push_back(terminal[output_layer[0]]);
        for (int i = 1; i < output_size; ++i) {
            output.push_back(terminal[output_layer[i]]);
        }
    } else {
        output.push_back(terminal[opt_layer[layer_number - 1]["name"]]);
    }
}

vtensorptr Neural_Network::Forward(Tensor *input_tensor_, bool train) {
    layer[0]->Forward(input_tensor_);
    for (int i = 1; i < layer_number; ++i) {
        layer[i]->Forward(train);
    }
    return output;
}

float Neural_Network::Backward(Tensor *target) {
    float loss = 0;
    if (model == "mtcnn") {
        float *target_ptr = target->weight;
        if (target_ptr[0] == 1) {
            Tensor cls_pos(1, 1, 1, 1);
            Tensor bbox_pos(1, 1, 4, 0);
            bbox_pos = {target_ptr[1], target_ptr[2], target_ptr[3], target_ptr[4]};
            layer[path[0][0]]->Backward(&cls_pos);
            layer[path[1][0]]->Backward(&bbox_pos);
            float loss_cls = layer[path[0][0]]->getLoss();
            float loss_bbox = layer[path[1][0]]->getLoss();
            if (loss_cls > loss_bbox) {
                for (int i = 1; i < path[0].size(); ++i) {
                    layer[path[0][i]]->Backward(&cls_pos);
                }
            }
            else {
                for (int i = 1; i < path[1].size(); ++i) {
                    layer[path[1][i]]->Backward(&bbox_pos);
                }
            }
            loss = loss_cls + loss_bbox * 0.5;
        }
        else if (target_ptr[0] == 0) {
            Tensor cls_neg(1, 1, 1, 0);
            for (int i = 0; i < path[0].size(); ++i) {
                layer[path[0][i]]->Backward(&cls_neg);
            }
        }
        else if (target_ptr[0] == -1) {
            Tensor bbox_part(1, 1, 4, 0);
            bbox_part = {target_ptr[1], target_ptr[2], target_ptr[3], target_ptr[4]};
            for (int i = 0; i < path[1].size(); ++i) {
                layer[path[1][i]]->Backward(&bbox_part);
            }
            loss *= 0.5;
        }
        else if (target_ptr[0] == -2) {
            Tensor landmark(1, 1, 10, 0);
            float *landmark_ptr = landmark.weight;
            for (int i = 5; i <= 14; ++i) {
                *(landmark_ptr++) = target_ptr[i];
            }
            for (int i = 0; i < path[2].size(); ++i) {
                layer[path[2][i]]->Backward(&landmark);
            }
            loss *= 0.5;
        }
    } else if (model == "yolov3" || model == "yolov4") {
        for (int i = layer_number; i--; ) {
            layer[i]->Backward(target);
            loss += layer[i]->getLoss();
        }
    } else {
        for (int i = layer_number; i--; ) {
            layer[i]->Backward(target);
            loss += layer[i]->getLoss();
        }
    }
    return loss;
}

vfloat Neural_Network::predict(Tensor *input) {
    vfloat output = Forward(input)[0]->toVector();
    float max_value = output[0];
    int max_index = 0, index, length = (int)output.size() / batch_size;
    for (index = 1; index < length; ++index) {
        if (output[index] > max_value) {
            max_value = output[index];
            max_index = index;
        }
    }
    return vfloat{static_cast<float>(max_index), max_value};
}

float Neural_Network::evaluate(vtensor &data_set, vtensor &target) {
    int correct = 0;
    int data_number = (int)data_set.size();
    vfloat check;
    for (int i = 0; i < data_number; ++i) {
        check = predict(&data_set[i]);
        if ((int)check[0] == (int)target[i].weight[0])
            correct++;
    }
    return (float)correct / data_number;
}

Neural_Network::Neural_Network(string model_) {
    model = model_;
    layer_number = 0;
    layer = nullptr;
    workspace = nullptr;
}

vector<Train_Args> Neural_Network::getTrainArgs() {
    vector<Train_Args> args_list;
    
    for (int i = 0; i < layer_number; ++i) {
        Train_Args args = layer[i]->getTrainArgs();
        if (args.valid)
            args_list.push_back(args);
    }
    return args_list;
}

void Neural_Network::alloc_workspace() {
    int max = 0, size;
    for (int i = 0; i < layer_number; ++i) {
        size = layer[i]->getWorkspaceSize();
        if (size > max)
            max = size;
    }
    if (max)
        workspace = new float [max];
    else
        workspace = nullptr;
//    printf("workspace: %lu bytes at ", max * sizeof(float));
//    cout << workspace << endl;
}

void Neural_Network::ClearGrad() {
    for (int i = 0; i < layer_number; ++i) {
        layer[i]->ClearGrad();
    }
}

Trainer::Trainer(Neural_Network *net, TrainerOption opt) {
    network = net;
    option = opt;
    learning_rate = (opt.find("learning_rate") == opt.end()) ? 0.001 : opt["learning_rate"];
    l1_decay = (opt.find("l1_decay") == opt.end()) ? 0.0 : opt["l1_decay"];
    l2_decay = (opt.find("l2_decay") == opt.end()) ? 0.0 : opt["l2_decay"];
    batch_size = (opt.find("batch_size") == opt.end()) ? network->getBatchSize() : opt["batch_size"];
    sub_division = (opt.find("sub_division") == opt.end()) ? 1 : opt["sub_division"];
//    batch_size /= sub_division;
    
    max_batches = (opt.find("max_batches") == opt.end()) ? 0 : opt["max_batches"];
    method = (opt.find("method") == opt.end()) ? SGD : (Trainer::Method)opt["method"];
    policy = (opt.find("policy") == opt.end()) ? CONSTANT : (Trainer::Policy)opt["policy"];
    momentum = (opt.find("momentum") == opt.end()) ? 0.9 : opt["momentum"];
    
    warmup = (opt.find("warmup") == opt.end()) ? 0 : opt["warmup"];
    power = (opt.find("power") == opt.end()) ? 4 : opt["power"];
    
    ro = (opt.find("ro") == opt.end()) ? 0.95 : opt["ro"];
    eps = (opt.find("eps") == opt.end()) ? 1e-6 : opt["eps"];
    beta_1 = (opt.find("beta_1") == opt.end()) ? 0.9 : opt["beta_1"];
    beta_2 = (opt.find("beta_2") == opt.end()) ? 0.999 : opt["beta_2"];
    
    step = (opt.find("step") == opt.end()) ? 1 : opt["step"];
    scale = (opt.find("scale") == opt.end()) ? 1 : opt["scale"];
    
    if (opt.find("steps") != opt.end()) {
        steps_num = opt["steps"];
        steps = Tensor(1, 1, steps_num, 0);
        scales = Tensor(1, 1, steps_num, 0);
        for (int i = 0; i < steps_num; ++i) {
            string steps_label = "steps_" + to_string(i + 1);
            string scales_label = "scales_" + to_string(i + 1);
            steps[i] = opt[steps_label];
            scales[i] = opt[scales_label];
        }
    }
    
    gamma = (opt.find("gamma") == opt.end()) ? 1 : opt["gamma"];
    
    seen = 0;
    batch_num = 0;
    args_num = 0;
    gsum.clear();
    xsum.clear();
}

float Trainer::get_learning_rate(){
    if (batch_num < warmup)
        return learning_rate * pow((float)batch_num / warmup, power);
    switch (policy) {
        case Policy::CONSTANT:
            return learning_rate;
        case STEP:
            return learning_rate * pow(scale, batch_num / step);
        case STEPS: {
            float rate = learning_rate;
            for(int i = 0; i < steps_num; ++i) {
                if(steps[i] > batch_num)
                    return rate;
                rate *= scales[i];
            }
            return rate;
        }
        case EXP:
            return learning_rate * pow(gamma, batch_num);
        case POLY:
            return learning_rate * pow(1 - (float)batch_num / max_batches, power);
        case RANDOM:
            return learning_rate * pow(Random(0, 1), power);
        case SIG:
            return learning_rate * (1.0 / (1.0 + exp(gamma * (batch_num - step))));
    }
    return 0;
}

vfloat Trainer::train(vtensor &data_set, vtensor &target_set, int epoch) {
    auto rng = std::mt19937((unsigned)time(NULL));
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
//            if (!(j % (data_set_size / 20))) printf("*");
            loss += train(data_set[index[j]], target_set[index[j]])[0];
        }
        printf("] ");
        printf("loss: %f\n", loss);
    }
    return vfloat{loss};
}

vfloat Trainer::train(Tensor &data, Tensor &target) {
    network->Forward(&data, true);
    network->ClearGrad();
    float loss = network->Backward(&target);
    float l1_decay_loss = 0;
    float l2_decay_loss = 0;
    ++seen;
    if ((seen % batch_size) == 0) {
        float current_rate = get_learning_rate();
        printf("learning_rate: %f batch_num: %d\n", current_rate, batch_num);
        ++batch_num;
        vector<Train_Args> args_list = network->getTrainArgs();
        
        vector<float*> weight_list; weight_list.reserve(args_num);
        vector<float*> grad_list; grad_list.reserve(args_num);
        vector<int> len_list; len_list.reserve(args_num);
        vector<vfloat> decay_list; decay_list.reserve(args_num);
        
        for (int i = (int)args_list.size(); i--; ) {
            Train_Args &args = args_list[i];
            
            Tensor* kernel_list = args.kernel;
            for (int j = args.kernel_list_size; j--; ) {
                Tensor &kernel = kernel_list[j];
                weight_list.push_back(kernel.weight);
                grad_list.push_back(kernel.delta_weight);
                len_list.push_back(args.kernel_size);
                decay_list.push_back(args.ln_decay_list);
            }
            Tensor* biases = args.biases;
            weight_list.push_back(biases->weight);
            grad_list.push_back(biases->delta_weight);
            len_list.push_back(args.biases_size);
            decay_list.push_back(args.ln_decay_list);
        }
        
        size_t len_list_size = len_list.size();
        if (gsum.empty() && (method != SGD || momentum > 0)) {
            for (int i = 0; i < len_list_size; ++i) {
                float *new_gsum = new float [len_list[i]]();
                //fill(new_gsum, new_gsum + len_list[i], 0);
                gsum.push_back(new_gsum);
                if (method == ADADELTA || method == ADAM) {
                    float *new_xsum = new float [len_list[i]]();
                    //fill(new_xsum, new_xsum + len_list[i], 0);
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
            float *gsumi = gsum[i];
            float *xsumi = xsum[i];
            
            float *weight_act = weight;
            float *grad_act = grad;
            
            float *gsum_act = gsumi;
            float *xsum_act = xsumi;
            
            for (int j = 0; j < len; ++j) {
                l1_decay_loss += l1_decay_local * abs(*weight_act);
                l2_decay_loss += l2_decay_local * *weight_act * *weight_act / 2;
                float l1_grad = l1_decay_local * (*weight_act > 0 ? 1 : -1);
                float l2_grad = l2_decay_local * *weight_act;
                
                float grad_ij = (l1_grad + l2_grad + *grad_act) / batch_size;
//                printf("grad: %f grad_ij: %f\n", *grad_act, grad_ij);
                
                if (method == ADADELTA) {
                    *gsum_act = ro * *gsum_act + (1 - ro) * grad_ij * grad_ij;
                    float delta_weight = -sqrt((*xsum_act + eps) / (*gsum_act + eps)) * grad_ij;
                    *xsum_act = ro * *xsum_act + (1 - ro) * delta_weight * delta_weight;
                    *weight_act += delta_weight;
                } else if (method == ADAM) {
                    *gsum_act = beta_1 * *gsum_act + (1 - beta_1) * grad_ij;
                    *xsum_act = beta_2 * *xsum_act + (1 - beta_2) * grad_ij * grad_ij;
                    float nor_gsumi = *gsum_act / (1 - pow(beta_1, (seen / batch_size) + 1));
                    float nor_xsumi = *xsum_act / (1 - pow(beta_2, (seen / batch_size) + 1));
                    float delta_weight = -((nor_gsumi) / (sqrt(nor_xsumi) + eps)) * current_rate;
                    *weight_act += delta_weight;
                } else { // SGD
                    if (momentum > 0) {
                        float delta_weight = momentum * *gsum_act - current_rate * grad_ij;
                        *gsum_act = delta_weight;
                        *weight_act += delta_weight;
//                        *gsum_act = momentum * *gsum_act + grad_ij;
//                        float delta_weight = - *gsum_act * current_rate;
//                        *weight_act += delta_weight;
                    } else {
                        *weight_act += -current_rate * grad_ij;
                    }
                }
                *grad_act = 0;
                
                weight_act++;
                grad_act++;
                gsum_act++;
                xsum_act++;
            }
        }
        if (!args_num)
            args_num = (int)len_list_size;
    }
    return vfloat{loss};
}

void Trainer::decade(float rate) {
    learning_rate *= rate;
}

vfloat Trainer::train_batch(vtensor &data_set, vtensor &target_set, int epoch) {
    auto rng = std::mt19937((unsigned)time(NULL));
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
        for (int j = 0; j + batch_size <= data_set_size; ) {
            Tensor data(1, 1, data_set[0].size * batch_size, 0);
            float *data_ptr = data.weight;
            Tensor label(1, 1, target_set[0].size * batch_size, 0);
            float *label_ptr = label.weight;
            for (int k = 0; k < batch_size; ++k, ++j) {
                Tensor &data_src = data_set[index[j]];
                float *data_src_ptr = data_src.weight;
                Tensor &label_src = target_set[index[j]];
                float *label_src_ptr = label_src.weight;
                for (int l = 0; l < data_set[0].size; ++l) {
                    *(data_ptr++) = *(data_src_ptr++);
                }
                for (int l = 0; l < target_set[0].size; ++l) {
                    *(label_ptr++) = *(label_src_ptr++);
                }
            }
            loss += train_batch(data, label)[0];
        }
        printf("] ");
        printf("loss: %f\n", loss);
    }
    return vfloat{loss};
}

vfloat Trainer::train_batch(Tensor &data, Tensor &target) {
    network->Forward(&data, true);
    network->ClearGrad();
    float loss = network->Backward(&target);
    float l1_decay_loss = 0;
    float l2_decay_loss = 0;
    seen += batch_size;
    if ((seen / batch_size) % sub_division == 0) {
        batch_num++;
        float current_rate = get_learning_rate();
//        printf("learning_rate: %f batch_num: %d\n", current_rate, batch_num);
        
        vector<Train_Args> args_list = network->getTrainArgs();
        
        vector<float*> weight_list;
        vector<float*> grad_list;
        vector<int> len_list;
        vector<vfloat> decay_list;
        
        for (int i = (int)args_list.size(); i--; ) {
            Train_Args &args = args_list[i];
            
            Tensor* kernel_list = args.kernel;
            for (int j = args.kernel_list_size; j--; ) {
                Tensor &kernel = kernel_list[j];
                weight_list.push_back(kernel.weight);
                grad_list.push_back(kernel.delta_weight);
                len_list.push_back(args.kernel_size);
                decay_list.push_back(args.ln_decay_list);
            }
            Tensor* biases = args.biases;
            weight_list.push_back(biases->weight);
            grad_list.push_back(biases->delta_weight);
            len_list.push_back(args.biases_size);
            decay_list.push_back(args.ln_decay_list);
        }
        
        size_t len_list_size = len_list.size();
        if (gsum.empty() && (method != SGD || momentum > 0)) {
            for (int i = 0; i < len_list_size; ++i) {
                float *new_gsum = new float [len_list[i]]();
                //fill(new_gsum, new_gsum + len_list[i], 0);
                gsum.push_back(new_gsum);
                if (method == ADADELTA || method == ADAM) {
                    float *new_xsum = new float [len_list[i]]();
                    //fill(new_xsum, new_xsum + len_list[i], 0);
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
            float *gsumi = gsum[i];
            float *xsumi = xsum[i];
            
            float *weight_act = weight;
            float *grad_act = grad;
            
            float *gsum_act = gsumi;
            float *xsum_act = xsumi;
            
            for (int j = 0; j < len; ++j) {
                l1_decay_loss += l1_decay_local * abs(*weight_act);
                l2_decay_loss += l2_decay_local * *weight_act * *weight_act / 2;
                float l1_grad = l1_decay_local * (*weight_act > 0 ? 1 : -1);
                float l2_grad = l2_decay_local * *weight_act;
                float grad_ij = (l1_grad + l2_grad + *grad_act) / batch_size;
                
                if (method == ADADELTA) {
                    *gsum_act = ro * *gsum_act + (1 - ro) * grad_ij * grad_ij;
                    float delta_weight = -sqrt((*xsum_act + eps) / (*gsum_act + eps)) * grad_ij;
                    *xsum_act = ro * *xsum_act + (1 - ro) * delta_weight * delta_weight;
                    *weight_act += delta_weight;
                } else if (method == ADAM) {
                    *gsum_act = beta_1 * *gsum_act + (1 - beta_1) * grad_ij;
                    *xsum_act = beta_2 * *xsum_act + (1 - beta_2) * grad_ij * grad_ij;
                    float nor_gsumi = *gsum_act / (1 - pow(beta_1, batch_num + 1));
                    float nor_xsumi = *xsum_act / (1 - pow(beta_2, batch_num + 1));
                    float delta_weight = -((nor_gsumi) / (sqrt(nor_xsumi) + eps)) * current_rate;
                    *weight_act += delta_weight;
                } else { // SGD
                    if (momentum > 0) {
//                        *gsum_act = momentum * *gsum_act - grad_ij;
//                        float delta_weight = - *gsum_act * current_rate;
//                        *weight_act += delta_weight;
                        float delta_weight = momentum * *gsum_act - current_rate * grad_ij;
                        *gsum_act = delta_weight;
                        *weight_act += delta_weight;
                    } else {
                        *weight_act += -current_rate * grad_ij;
                    }
                }
                *grad_act = 0;
                
                weight_act++;
                grad_act++;
                gsum_act++;
                xsum_act++;
            }
        }
    }
    return vfloat{loss};
}
