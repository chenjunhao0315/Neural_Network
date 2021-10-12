//
//  Layer.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include "Layer.hpp"

static ParameterParser layerparameter("layer.txt");

BaseLayer::~BaseLayer() {
    if (output_tensor)
        delete output_tensor;
    if (kernel)
        delete [] kernel;
    if (biases)
        delete biases;
}

BaseLayer::BaseLayer() {
    input_tensor = nullptr;
    output_tensor = nullptr;
    kernel = nullptr;
    biases = nullptr;
    memset(&info, 0, sizeof(Info));
}

BaseLayer::BaseLayer(const BaseLayer &L) {
    if (this != &L) {
        type = L.type;
        name = L.name;
        input_name = L.input_name;
        info = L.info;
        opt = L.opt;
        input_tensor = L.input_tensor;
        if (L.output_tensor)
            output_tensor = new Tensor(L.output_tensor);
        else
            output_tensor = nullptr;
        if (L.kernel) {
            kernel = new Tensor [info.kernel_num];
            for (int i = 0; i < info.kernel_num; ++i) {
                kernel[i] = L.kernel[i];
            }
        } else {
            kernel = nullptr;
        }
        if (L.biases) {
            biases = new Tensor(L.biases);
        } else {
            biases = nullptr;
        }
    }
}

BaseLayer::BaseLayer(BaseLayer &&L) {
    type = L.type;
    name = L.name;
    input_name = L.input_name;
    info = L.info;
    opt = L.opt;
    input_tensor = L.input_tensor;
    L.input_tensor = nullptr;
    output_tensor = L.output_tensor;
    L.output_tensor = nullptr;
    kernel = L.kernel;
    L.kernel = nullptr;
    biases = L.biases;
    L.biases = nullptr;
}

BaseLayer& BaseLayer::operator=(const BaseLayer &L) {
    if (this != &L) {
        type = L.type;
        name = L.name;
        input_name = L.input_name;
        info = L.info;
        opt = L.opt;
        input_tensor = L.input_tensor;
        if (L.output_tensor)
            output_tensor = new Tensor(L.output_tensor);
        else
            output_tensor = nullptr;
        if (L.kernel) {
            kernel = new Tensor [info.kernel_num];
            for (int i = 0; i < info.kernel_num; ++i) {
                kernel[i] = L.kernel[i];
            }
        } else {
            kernel = nullptr;
        }
        if (L.biases) {
            biases = new Tensor(L.biases);
        } else {
            biases = nullptr;
        }
    }
    return *this;
}

int BaseLayer::getWorkspaceSize() {
    switch(type) {
        case LayerType::Convolution:
            return info.workspace_size;
        default:
            return 0;
    }
    return 0;
}

Train_Args BaseLayer::getTrainArgs() {
    switch(type) {
        case LayerType::Convolution:
            return Train_Args(kernel, biases, info.kernel_width * info.kernel_height * info.input_dimension * info.output_dimension, 1, (info.batchnorm) ? 0 : info.output_dimension, vfloat{0, 1});
        case LayerType::FullyConnected:
            return Train_Args(kernel, biases, info.input_number * info.output_dimension, 1, (info.batchnorm) ? 0 : info.output_dimension, vfloat{0, 1});
        case LayerType::PRelu:
            return Train_Args(kernel, kernel, info.input_number, 1, 0, vfloat{0, 0});
        case LayerType::BatchNormalization:
            return Train_Args(kernel, biases, info.output_dimension, 3, info.output_dimension, vfloat{0, 0});
        default:
            return Train_Args();
    }
    return Train_Args();
}

string BaseLayer::type_to_string() {
    switch(type) {
        case Input: return "Input";
        case FullyConnected: return "FullyConnected";
        case Relu: return "Relu";
        case PRelu: return "PRelu";
        case Softmax: return "Softmax";
        case Convolution: return "Convolution";
        case Pooling: return "Pooling";
        case EuclideanLoss: return "EuclideanLoss";
        case ShortCut: return "ShortCut";
        case LRelu: return "LRelu";
        case Sigmoid: return "Sigmoid";
        case BatchNormalization: return "BatchNorm";
        case UpSample: return "UpSample";
        case Concat: return "Concat";
        case Yolov3: return "YOLOv3";
        case Yolov4: return "YOLOv4";
        case Mish: return "Mish";
        case Swish: return "Swish";
        case Dropout: return "Dropout";
        case AvgPooling: return "AvgPooling";
        case Error: return "Error";
    }
    return "Unknown";
}

Tensor* BaseLayer::connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace) {
    input_tensor = input_tensor_;
    return output_tensor;
}

void BaseLayer::shape() {
    printf("%-17s%-13s %-13s  ", type_to_string().c_str(), name.c_str(), input_name.c_str());
    printf("(%d * %d * %d)\n", info.output_width, info.output_height, info.output_dimension);
    //    if (type == LayerType::Convolution || type == LayerType::FullyConnected) {
    //        for (int i = 0; i < info.output_dimension; ++i) {
    //            printf("Weight:\n%d: ", i);
    //            kernel[i].showWeight();
    //        }
    //        printf("Bias:\n");
    //        biases->showWeight();
    //    } else if (type == LayerType::PRelu) {
    //        kernel[0].showWeight();
    //    } else if (type == LayerType::BatchNormalization) {
    //        printf("Scale:\n");
    //        kernel[0].showWeight();
    //        printf("Running mean:\n");
    //        kernel[1].showWeight();
    //        printf("Running variance:\n");
    //        kernel[2].showWeight();
    //        printf("Bias:\n");
    //        biases->showWeight();
    //    }
}

void BaseLayer::show_detail() {
    switch(type) {
        case LayerType::Convolution: {
            printf("%-13s", name.c_str());
            printf("%4d ", info.output_dimension);
            printf("%2d x%2d /%2d (%2d) ", info.kernel_width, info.kernel_height, info.stride, info.padding);
            printf("%4d x%4d x%4d ->%4d x%4d x%4d ", info.input_width, info.input_height, info.input_dimension, info.output_width, info.output_height, info.output_dimension);
            long operations = getOperations();
            if (operations < 10000) {
                printf("%5.3f KFlOPs", operations / 1000.0);
            } else if (operations < 10000000) {
                printf("%5.3f MFlOPs", operations / 1000000.0);
            } else {
                printf("%5.3f BFlOPs", operations / 1000000000.0);
            }
            printf("\n");
            break;
        }
        case LayerType::Pooling: {
            printf("%-13s", name.c_str());
            printf("     ");
            printf("%2d x%2d /%2d (%2d) ", info.kernel_width, info.kernel_height, info.stride, info.padding);
            printf("\n");
            break;
        }
        case LayerType::UpSample: {
            printf("%-13s", name.c_str());
            printf("     ");
            printf("%2d x%2d /%2d (%2d) ", 1, 1, info.stride, info.padding);
            printf("\n");
            break;
        }
        case LayerType::Concat: {
            printf("%-13s", name.c_str());
            printf("     ");
            printf("%2d /%2d          ", info.splits, info.split_id);
            printf("%4d x%4d x%4d ->%4d x%4d x%4d ", info.input_width, info.input_height, info.input_dimension, info.output_width, info.output_height, info.output_dimension);
            printf("%s", (opt.find("concat") == opt.end()) ? "" : opt["concat"].c_str());
            printf("\n");
            break;
        }
        default:
            break;
    }
}

long BaseLayer::getOperations() {
    switch(type) {
        case LayerType::Convolution: {
            return 2l * info.output_dimension * info.kernel_width * info.kernel_height * info.input_dimension * info.output_width * info.output_height;
        }
        case LayerType::FullyConnected: {
            return 2l * info.input_number * info.output_number;
        }
        default:
            return 0;
    }
    return 0;
}

int BaseLayer::getParameter(int type) {
    switch (type) {
        case 0: return info.output_width; break;
        case 1: return info.output_height; break;
        case 2: return info.output_dimension; break;
        default: break;
    }
    return 0;
}

void BaseLayer::ClearGrad() {
    output_tensor->clearDeltaWeight();
}

bool BaseLayer::save_raw(FILE *f) {
    if (type == LayerType::FullyConnected || type == LayerType::Convolution) {
        if (!info.batchnorm) {
            biases->save_raw(f);
        }
        kernel[0].save_raw(f);
    } else if (type == LayerType::PRelu) {
        kernel->save_raw(f);
    } else if (type == LayerType::BatchNormalization) {
        biases->save_raw(f);
        kernel[0].save_raw(f);
        kernel[3].save_raw(f);
        kernel[4].save_raw(f);
    }
    return true;
}

bool BaseLayer::load_raw(FILE *f) {
    if (type == LayerType::FullyConnected || type == LayerType::Convolution) {
        if (!info.batchnorm) {
            biases->load_raw(f);
        }
        kernel[0].load_raw(f);
    } else if (type == LayerType::PRelu) {
        kernel->load_raw(f);
    } else if (type == LayerType::BatchNormalization) {
        biases->load_raw(f);
        kernel[0].load_raw(f);
        kernel[3].load_raw(f);
        kernel[4].load_raw(f);
        // Test
        kernel[3].copyTo(kernel[1]);
        kernel[4].copyTo(kernel[2]);
    }
    return true;
}

bool BaseLayer::to_prototxt(FILE *f, int refine_id, vector<LayerOption> &refine_struct, unordered_map<string, int> &id_table) {
    if (type == LayerType::Input) {
        fprintf(f, "input: \"data\"\n");
        fprintf(f, "input_dim: %d\n", info.batch_size);
        fprintf(f, "input_dim: %d\n", info.output_dimension);
        fprintf(f, "input_dim: %d\n", info.output_width);
        fprintf(f, "input_dim: %d\n", info.output_height);
    } else {
        fprintf(f, "layer {\n");
        fprintf(f, "  name: \"%s\"\n", name.c_str());
        fprintf(f, "  type: \"%s\"\n", type_to_string().c_str());
        fprintf(f, "  bottom: \"%s\"\n", refine_struct[refine_id]["input_name"].c_str());
        fprintf(f, "  top: \"%s\"\n", refine_struct[refine_id]["name"].c_str());
        
        if (type == LayerType::FullyConnected) {
            fprintf(f, "  connected_param {\n");
            fprintf(f, "    num_output: %d\n", info.output_dimension);
            fprintf(f, "  }\n");
        } else if (type == LayerType::Softmax) {
        } else if (type == LayerType::Convolution) {
            fprintf(f, "  convolution_param {\n");
            fprintf(f, "    num_output: %d\n", info.output_dimension);
            fprintf(f, "    kernel_size: %d\n", info.kernel_width);
            fprintf(f, "    pad: %d\n", info.padding);
            fprintf(f, "    stride: %d\n", info.stride);
            if (info.batchnorm) {
                fprintf(f, "    bias_term: false\n");
            } else {
                fprintf(f, "    bias_term: true\n");
            }
            fprintf(f, "  }\n");
        } else if (type == LayerType::Relu) {
        } else if (type == LayerType::Pooling) {
            fprintf(f, "  pool_param {\n");
            fprintf(f, "    pool: MAX\n");
            fprintf(f, "    kernel_size: %d\n", info.kernel_width);
            fprintf(f, "    stride: %d\n", info.stride);
            fprintf(f, "  }\n");
        } else if (type == LayerType::EuclideanLoss) {
        } else if (type == LayerType::PRelu) {
        } else if (type == LayerType::ShortCut) {
            fprintf(f, "  bottom: \"%s\"\n", refine_struct[id_table[opt["shortcut"]]]["name"].c_str());
        } else if (type == LayerType::LRelu) {
            fprintf(f, "  lrelu_param {\n");
            fprintf(f, "    negative_slope: %g\n", kernel[0][0]);
            fprintf(f, "  }\n");
        } else if (type == LayerType::Sigmoid) {
        } else if (type == LayerType::BatchNormalization) {
        } else if (type == LayerType::UpSample) {
        } else if (type == LayerType::Concat) {
            for (int i = 1; i <= info.concat_num; ++i) {
                fprintf(f, "  bottom: \"%s\"\n", refine_struct[id_table[opt["concat_" + to_string(i) + "_name"]]]["name"].c_str());
            }
            fprintf(f, "  concat_param {\n");
            fprintf(f, "    splits: %d\n", info.splits);
            fprintf(f, "    split_id: %d\n", info.split_id);
            fprintf(f, "  }\n");
        } else if (type == LayerType::Yolov3) {
            fprintf(f, "  yolov3_param {\n");
            fprintf(f, "    classes: %d\n", info.classes);
            fprintf(f, "    total_anchor_num: %d\n", info.total_anchor_num);
            fprintf(f, "    anchor_num: %d\n", info.anchor_num);
            fprintf(f, "    anchor: \"%s\"\n", opt["anchor"].c_str());
            fprintf(f, "    mask: \"%s\"\n", opt["mask"].c_str());
            fprintf(f, "  }\n");
        } else if (type == LayerType::Yolov4) {
            fprintf(f, "  yolov4_param {\n");
            fprintf(f, "    classes: %d\n", info.classes);
            fprintf(f, "    total_anchor_num: %d\n", info.total_anchor_num);
            fprintf(f, "    anchor_num: %d\n", info.anchor_num);
            fprintf(f, "    anchor: \"%s\"\n", opt["anchor"].c_str());
            fprintf(f, "    mask: \"%s\"\n", opt["mask"].c_str());
            fprintf(f, "  }\n");
        } else if (type == LayerType::Mish) {
        } else if (type == LayerType::Swish) {
        } else if (type == LayerType::Dropout) {
            fprintf(f, "  dropout_param {\n");
            fprintf(f, "    ratioi: %g\n", info.probability);
            fprintf(f, "  }\n");
        }
        fprintf(f, "}\n");
    }
    return true;
}

ParameterParser::ParameterParser(const char *param_filename) {
    fstream param_file;
    param_file.open(param_filename);
    
    while (!param_file.eof()) {
        string param;
        param_file >> param;
        param_list.push_back(param);
    }
    
    param_file.close();
    this->parse();
}

void ParameterParser::parse() {
    for (int i = 0; i < param_list.size(); ) {
        CHECK_IF_QUIT(param_list[i], "{");
        Parameter param(param_list[i++]);
        CHECK_IFNOT_QUIT(param_list[i++], "{");
        for ( ; param_list[i] != "}" && i < param_list.size(); ) {
            if (param_list[i] == "REQUIRED") {
                param.addParameter(ParameterData(REQUIRED, param_list[i + 1], param_list[i + 2], ""));
                i += 3;
            } else if (param_list[i] == "OPTION") {
                param.addParameter(ParameterData(OPTION, param_list[i + 1], param_list[i + 2], param_list[i + 3]));
                i += 4;
            } else {
                fprintf(stderr, "[ParameterParser] Syntax error!\n"); exit(1);
            }
        }
        CHECK_IFNOT_QUIT(param_list[i++], "}");
        LayerParameter::AddParameter(param);
    }
}


InputLayer::InputLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Input;
    name = (opt.find("name") == opt.end()) ? "in" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_dimension = (opt.find("input_dimension") == opt.end()) ? atoi(opt["output_dimension"].c_str()) : atoi(opt["input_dimension"].c_str());
    info.output_width = (opt.find("input_width") == opt.end()) ? atoi(opt["output_width"].c_str()) : atoi(opt["input_width"].c_str());
    info.output_height = (opt.find("input_height") == opt.end()) ? atoi(opt["output_height"].c_str()) : atoi(opt["input_height"].c_str());
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(Input);

void InputLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    copy_cpu(info.output_number * info.batch_size, input_tensor->weight, output_tensor->weight);
}

ConvolutionLayer::ConvolutionLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Convolution;
    name = (opt.find("name") == opt.end()) ? "conv" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.kernel_width = atoi(opt["kernel_width"].c_str());
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    info.kernel_height = (opt.find("kernel_height") == opt.end()) ? info.kernel_width : atoi(opt["kernel_height"].c_str());
    info.stride = (opt.find("stride") == opt.end()) ? 1 : atoi(opt["stride"].c_str());
    info.padding = (opt.find("padding") == opt.end()) ? 0 : ((opt["padding"] == "same") ? ((info.kernel_width - 1) / 2) : atoi(opt["padding"].c_str()));
    
    info.output_width = (info.input_width + info.padding * 2 - info.kernel_width) / info.stride + 1;
    info.output_height = (info.input_height + info.padding * 2 - info.kernel_height) / info.stride + 1;
    info.output_dimension = atoi(opt["number_kernel"].c_str());
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    info.batchnorm = (opt.find("batchnorm") != opt.end());
    
    
    float bias = (opt.find("bias") == opt.end()) ? 0 : atof(opt["bias"].c_str());
    
    kernel = new Tensor [1];
    kernel[0] = Tensor(1, 1, info.kernel_width * info.kernel_height * info.input_dimension * info.output_dimension, 0);
    kernel[0].extend();
    float *kernel_weight = kernel->weight;
    float scale = sqrt(2.0 / (info.kernel_width * info.kernel_height * info.input_dimension));
    for (int i = 0; i < info.kernel_width * info.kernel_height * info.input_dimension * info.output_dimension; ++i) {
        *(kernel_weight++) = randn(0.0, scale);
    }
    info.kernel_num = 1;
    
    biases = new Tensor(1, 1, info.output_dimension, bias);
    biases->extend();
    
    int kernel_n = info.kernel_width * info.kernel_height * info.input_dimension;
    int out_size = info.output_width * info.output_height;
    int workspace_size = out_size * kernel_n;
    info.workspace_size = workspace_size;
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(Convolution);

Tensor* ConvolutionLayer::connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace_) {
    input_tensor = input_tensor_;
    workspace = workspace_;
    return output_tensor;
}

void ConvolutionLayer::Forward(bool train) {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    float *weights = kernel->weight;
    float *bias = biases->weight;
    
    int batch_size = info.batch_size;
    int input_number = info.input_number;
    int output_number = info.output_number;
    int output_size = info.output_width * info.output_height;
    int kernel_number = info.kernel_width * info.kernel_height * info.input_dimension;
    int kernel_size = info.kernel_width;
    
    int input_width = info.input_width;
    int input_height = info.input_height;
    int input_dimension = info.input_dimension;
    
    int stride = info.stride;
    int padding = info.padding;
    
    fill_cpu(output_number * batch_size, output, 0);
    
    int m = info.output_dimension;
    int k = kernel_number;
    int n = output_size;
    for(int i = 0; i < batch_size; ++i){
        float *a = weights;
        float *b = workspace;
        float *c = output + i * n * m;
        float *im = input + i * input_number;
        
        if (kernel_size == 1) {
            // Doesn't need to rearrange
            b = im;
        } else {
            // Rearrange input tensor and put it into workspace
            im2col_cpu(im, input_dimension, input_height, input_width, kernel_size, stride, padding, b);
        }
        // Matrix multiplication (input with kernel) (Both not transpose)
        // Calculate output (input_channel * kernel_n) dot (kernel_n * output_channel)
        gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
    }
    
    if(!info.batchnorm){
        // If not batchnorm, add bias
        add_bias(output, bias, batch_size, info.output_dimension, output_size);
    }
}

void ConvolutionLayer::Backward(Tensor *none) {
    float *input = input_tensor->weight;
    float *input_delta = input_tensor->delta_weight;
    float *output_delta = output_tensor->delta_weight;
    float *bias_delta = biases->delta_weight;
    float *weights = kernel->weight;
    float *weights_delta = kernel->delta_weight;
    
    int batch_size = info.batch_size;
    int kernel_size = info.kernel_width;
    int stride = info.stride;
    int padding = info.padding;
    
    int input_width = info.input_width;
    int input_height = info.input_height;
    int input_dimension = info.input_dimension;
    int input_number = info.input_number;
    
    
    int m = info.output_dimension;
    int n = info.kernel_width * info.kernel_height * info.input_dimension;
    int k = info.output_width * info.output_height;
    
    if(!info.batchnorm){
        // If not batchnorm, backpropagate bias
        backward_bias(bias_delta, output_delta, batch_size, info.output_dimension, k);
    }
    
    for(int i = 0; i < batch_size; ++i){
        float *a = output_delta + i * m * k;
        float *b = workspace;
        float *c = weights_delta;
        
        float *im  = input + i * input_number;
        float *imd = input_delta + i * input_number;
        
        if(kernel_size == 1){
            // Doesn't need to rearrange
            b = im;
        } else {
            // Rearrange input tensor and put it into workspace
            im2col_cpu(im, input_dimension, input_height, input_width, kernel_size, stride, padding, b);
        }
        // Matrix multiplication (input with output_delta) (input transpose)
        // Calculate weight_delta
        gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
        a = weights;
        b = output_delta + i * m * k;
        c = workspace;
        if (kernel_size == 1) {
            c = imd;
        }
        // Matrix multiplication (weight with output_delta) (weight transpose)
        // Calculate input_delta
        gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);
        if (kernel_size != 1) {
            // Restore arrangement
            col2im_cpu(workspace, input_dimension, input_height, input_width, kernel_size, stride, padding, imd);
        }
    }
}

PoolingLayer::PoolingLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Pooling;
    name = (opt.find("name") == opt.end()) ? "pool" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.kernel_width = atoi(opt["kernel_width"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    info.kernel_height = (opt.find("kernel_height") == opt.end()) ? info.kernel_width : atoi(opt["kernel_height"].c_str());
    info.stride = (opt.find("stride") == opt.end()) ? 1 : atoi(opt["stride"].c_str());
    info.padding = (opt.find("padding") == opt.end()) ? 0 : ((opt["padding"] == "same") ? ((info.kernel_width - 1)) : atoi(opt["padding"].c_str()));
    
    info.output_width = (info.input_width + info.padding - info.kernel_width) / info.stride + 1;
    info.output_height = (info.input_height + info.padding - info.kernel_height) / info.stride + 1;
    info.output_dimension = info.input_dimension;
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    
    kernel = new Tensor [1];
    kernel[0] = Tensor(1, 1, info.output_number * info.batch_size, 0);
    info.kernel_num = 1;
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(Pooling);

void PoolingLayer::Forward(bool train) {
    int stride = info.stride;
    
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    
    int input_width = info.input_width;
    int input_height = info.input_height;
    int input_dimension = info.input_dimension;
    int kernel_size = info.kernel_width;
    
    int w_offset = -info.padding / 2;
    int h_offset = -info.padding / 2;
    
    int h = info.output_height;
    int w = info.output_width;
    int c = info.output_dimension;
    
    float *indexes = kernel->weight;
    
    for(int b = 0; b < info.batch_size; ++b){
        for(int k = 0; k < c; ++k){
            for(int i = 0; i < h; ++i){
                for(int j = 0; j < w; ++j){
                    int out_index = j + w * (i + h * (k + c * b));
                    float max = -10000000;
                    int max_i = -1;
                    for(int n = 0; n < kernel_size; ++n){
                        for(int m = 0; m < kernel_size; ++m){
                            int cur_h = h_offset + i * stride + n;
                            int cur_w = w_offset + j * stride + m;
                            int index = cur_w + input_width * (cur_h + input_height * (k + b * input_dimension));
                            bool valid = (cur_h >= 0 && cur_h < input_height && cur_w >= 0 && cur_w < input_width);
                            float val = (valid != false) ? input[index] : -10000000;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    output[out_index] = max;
                    indexes[out_index] = max_i;
                }
            }
        }
    }
}

void PoolingLayer::Backward(Tensor *none) {
    float *output_delta = output_tensor->delta_weight;
    float *input_delta = input_tensor->delta_weight;
    float *indexes = kernel->weight;
    
    int total_size = info.output_width * info.output_height * info.output_dimension * info.batch_size;
    for(int i = 0; i < total_size; ++i){
        int index = indexes[i];
        input_delta[index] += output_delta[i];
    }
}

FullyConnectedLayer::FullyConnectedLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::FullyConnected;
    name = (opt.find("name") == opt.end()) ? "fc" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    info.output_width = 1;
    info.output_height = 1;
    info.output_dimension = atoi(opt["number_neurons"].c_str());
    info.output_number = info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    info.batchnorm = (opt.find("batchnorm") != opt.end());
    
    kernel = new Tensor [1];
    kernel[0] = Tensor(info.input_number, 1, info.output_dimension, 0);
    kernel[0].extend();
    float *kernel_weight = kernel->weight;
    float scale = sqrt(1.0 / (info.input_number));
    for (int i = 0; i < info.input_number * info.output_dimension; ++i) {
        *(kernel_weight++) = randn(0.0, scale);
    }
    info.kernel_num = 1;
    
    float bias = (opt.find("bias") == opt.end()) ? 0 : atof(opt["bias"].c_str());
    biases = new Tensor(1, 1, info.output_dimension, bias);
    biases->extend();
    output_tensor = new Tensor(1, 1, info.output_dimension * info.batch_size, 0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(FullyConnected);

void FullyConnectedLayer::Forward(bool train) {
    fill_cpu(info.output_dimension * info.batch_size, output_tensor->weight, 0);
    
    int m = info.batch_size;
    int k = info.input_number;
    int n = info.output_number;
    float *a = input_tensor->weight;
    float *b = kernel->weight;
    float *c = output_tensor->weight;
    gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
    if(!info.batchnorm){
        add_bias(output_tensor->weight, biases->weight, info.batch_size, info.output_dimension, 1);
    }
}

void FullyConnectedLayer::Backward(Tensor *none) {
    if(!info.batchnorm){
        backward_bias(biases->delta_weight, output_tensor->delta_weight, info.batch_size, info.output_dimension, 1);
    }
    
    int m = info.output_dimension;
    int k = info.batch_size;
    int n = info.input_number;
    float *a = output_tensor->delta_weight;
    float *b = input_tensor->weight;
    float *c = kernel->delta_weight;
    gemm(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);
    
    m = info.batch_size;
    k = info.output_dimension;
    n = info.input_number;
    
    a = output_tensor->delta_weight;
    b = kernel->weight;
    c = input_tensor->delta_weight;
    
    gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
}

ReluLayer::ReluLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Relu;
    name = (opt.find("name") == opt.end()) ? "relu" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = atoi(opt["input_width"].c_str());
    info.output_height = atoi(opt["input_height"].c_str());
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.output_width * info.output_height * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(Relu);

void ReluLayer::Forward(bool train) {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    float value;
    
    for (int i = info.input_number * info.batch_size; i--; ) {
        value = *(input++);
        *(output++) = (value < 0) ? 0 : value;
    }
}

void ReluLayer::Backward(Tensor *none) {
    int one_batch_size = info.input_number;
    float *output = output_tensor->weight;
    float *output_delta = output_tensor->delta_weight;
    float *input_delta = input_tensor->delta_weight;
    
    for (int b = info.batch_size; b--; ) {
        for (int i = info.input_number; i--; ) {
            if (output[i] > 0)
                input_delta[i] += output_delta[i];
        }
        output += one_batch_size;
        output_delta += one_batch_size;
        input_delta += one_batch_size;
    }
}

PReluLayer::PReluLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::PRelu;
    name = (opt.find("name") == opt.end()) ? "prelu" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = atoi(opt["input_width"].c_str());
    info.output_height = atoi(opt["input_height"].c_str());
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.output_width * info.output_height * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    float alpha = (opt.find("alpha") == opt.end()) ? 0.25 : atof(opt["alpha"].c_str());
    kernel = new Tensor [1];
    kernel[0] = Tensor(info.output_width, info.output_height, info.output_dimension, alpha);
    kernel[0].extend();
    info.kernel_num = 1;
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(PRelu);

void PReluLayer::Forward(bool traiin) {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    
    for (int b = info.batch_size; b--; ) {
        float *alpha = kernel->weight;
        float value;
        for (int i = info.input_number; i--; ++alpha) {
            value = *(input++);
            *(output++) = (value < 0) ? value * *alpha : value;
        }
    }
}

void PReluLayer::Backward(Tensor *none) {
    int one_batch_size = info.input_number;
    float *output = output_tensor->weight;
    float *output_delta = output_tensor->delta_weight;
    float *input_delta = input_tensor->delta_weight;
    
    for (int b = info.batch_size; b--; ) {
        float *alpha = kernel->weight;
        float *alpha_delta = kernel->delta_weight;
        float chain_grad;
        
        for (int i = 0; i < info.input_number; ++i) {
            chain_grad = output_delta[i];
            if (output[i] < 0) {
                input_delta[i] += chain_grad * alpha[i];
                alpha_delta[i] += output[i] * chain_grad;
            }
            else {
                input_delta[i] += chain_grad;
                alpha_delta[i] += 0;
            }
        }
        output += one_batch_size;
        output_delta += one_batch_size;
        input_delta += one_batch_size;
    }
}

SoftmaxLayer::SoftmaxLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Softmax;
    name = (opt.find("name") == opt.end()) ? "sm" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = 1;
    info.output_height = 1;
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = atoi(opt["input_width"].c_str()) * atoi(opt["input_height"].c_str()) * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    kernel = new Tensor [1];
    kernel[0] = Tensor(1, 1, info.output_dimension * info.batch_size, 0);
    info.kernel_num = 1;
    
    output_tensor = new Tensor(1, 1, info.output_dimension * info.batch_size, 0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(Softmax);

void SoftmaxLayer::Forward(bool train) {
    int one_batch_input_size = info.input_number;
    int one_batch_output_size = info.output_dimension;
    
    float *input = input_tensor->weight;
    float *expo_sum = kernel->weight;
    float *output = output_tensor->weight;
    
    for (int b = info.batch_size; b--; ) {
        float max = input[0];
        for (int i = 1; i < info.output_dimension; ++i) {
            if (input[i] > max)
                max = input[i];
        }
        float sum = 0;
        for (int i = 0; i < info.output_dimension; ++i) {
            sum += (expo_sum[i] = exp(input[i] - max));
        }
        for (int i = 0; i < info.output_dimension; ++i) {
            expo_sum[i] /= sum;
            output[i] = expo_sum[i];
        }
        input += one_batch_input_size;
        expo_sum += one_batch_output_size;
        output += one_batch_output_size;
    }
}

void SoftmaxLayer::Backward(Tensor *target) {
    int one_batch_input_size = info.input_number;
    int one_batch_output_size = info.output_dimension;
    
    float *input_delta = input_tensor->delta_weight;
    float *expo_sum = kernel->weight;
    float *target_ptr = target->weight;
    
    int output_dimension = info.output_dimension;
    
    for (int b = 0; b < info.batch_size; ++b) {
        for (int i = 0; i < output_dimension; ++i) {
            float indicator = (i == target_ptr[b]) ? 1.0 : 0.0;
            float mul = -(indicator - expo_sum[i]);
            input_delta[i] = mul;
        }
        input_delta += one_batch_input_size;
        expo_sum += one_batch_output_size;
    }
    float &loss = info.loss; loss = 0;
    expo_sum = kernel->weight;
    for (int b = 0; b < info.batch_size; ++b) {
        loss += -log(expo_sum[(int)target_ptr[b]]);
        expo_sum += one_batch_output_size;
    }
}

EuclideanLossLayer::EuclideanLossLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::EuclideanLoss;
    name = (opt.find("name") == opt.end()) ? "el" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = atoi(opt["input_width"].c_str());
    info.output_height = atoi(opt["input_height"].c_str());
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(EuclideanLoss);

void EuclideanLossLayer::Forward(bool train) {
    copy_cpu(info.output_number * info.batch_size, input_tensor->weight, output_tensor->weight);
}

void EuclideanLossLayer::Backward(Tensor *target) {
    int one_batch_size = info.output_dimension;
    
    float *input = input_tensor->weight;
    float *input_delta = input_tensor->delta_weight;
    float *target_ptr = target->weight;
    float &loss = info.loss; loss = 0;
    
    for (int b = 0; b < info.batch_size; ++b) {
        for (int i = 0; i < info.output_dimension; ++i) {
            float delta = input[i] - target_ptr[i + b * one_batch_size];
            input_delta[i] += delta;
            loss += 0.5 * delta * delta;
        }
        input += one_batch_size;
        input_delta += one_batch_size;
    }
}

ShortCutLayer::ShortCutLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::ShortCut;
    name = (opt.find("name") == opt.end()) ? "sc" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = atoi(opt["input_width"].c_str());
    info.output_height = atoi(opt["input_height"].c_str());
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.output_number = info.input_number = info.output_width * info.output_height * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    info.shortcut_width = atoi(opt["shortcut_width"].c_str());
    info.shortcut_height = atoi(opt["shortcut_height"].c_str());
    info.shortcut_dimension = atoi(opt["shortcut_dimension"].c_str());
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(ShortCut);

Tensor* ShortCutLayer::connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace) {
    input_tensor = input_tensor_;
    shortcut_tensor = extra_tensor_[0];
    return output_tensor;
}

void ShortCutLayer::Forward(bool train) {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    float *shortcut = shortcut_tensor->weight;
    
    int shortcut_width = info.shortcut_width;
    int shortcut_height = info.shortcut_height;
    int shortcut_dimension = info.shortcut_dimension;
    
    copy_cpu(info.input_number * info.batch_size, input, output);
    if (shortcut_width == info.output_width && shortcut_height == info.output_height && shortcut_dimension == info.output_dimension) {
        for (int i = info.output_number * info.batch_size; i--; ) {
            *(output++) = *(input++) + *(shortcut++);
        }
    } else {
        shortcut_cpu(info.batch_size, shortcut_width, shortcut_height, shortcut_dimension, shortcut, info.output_width, info.output_height, info.output_dimension, 1, 1, output);
    }
}

void ShortCutLayer::Backward(Tensor *none) {
    float *input_delta = input_tensor->delta_weight;
    float *output_delta = output_tensor->delta_weight;
    for (int i = info.input_number * info.batch_size; i--; ) {
        *(input_delta++) += *(output_delta++);
    }
    
    int shortcut_width = info.shortcut_width;
    int shortcut_height = info.shortcut_height;
    int shortcut_dimension = info.shortcut_dimension;
    
    float *shortcut_delta = shortcut_tensor->delta_weight;
    output_delta = output_tensor->delta_weight;
    
    shortcut_cpu(info.batch_size, info.output_width, info.output_height, info.output_dimension, output_delta, shortcut_width, shortcut_height, shortcut_dimension, 1, 1, shortcut_delta);
}

void ShortCutLayer::shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out) {
    int stride = w1 / w2;
    int sample = w2 / w1;
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;
    
    for(int b = 0; b < batch; ++b){
        for(int k = 0; k < minc; ++k){
            for(int j = 0; j < minh; ++j){
                for(int i = 0; i < minw; ++i){
                    int out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
                    int add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));
                    out[out_index] = s1 * out[out_index] + s2 * add[add_index];
                }
            }
        }
    }
}

LReluLayer::LReluLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::LRelu;
    name = (opt.find("name") == opt.end()) ? "lrelu" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = atoi(opt["input_width"].c_str());
    info.output_height = atoi(opt["input_height"].c_str());
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.output_width * info.output_height * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    float alpha = (opt.find("alpha") == opt.end()) ? 0.1 : atof(opt["alpha"].c_str());
    kernel = new Tensor [1];
    kernel[0] = Tensor(1, 1, 1, alpha);
    info.kernel_num = 1;
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(LRelu);

void LReluLayer::Forward(bool train) {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    float value, scale = kernel[0][0];
    for (int i = info.input_number * info.batch_size; i--; ++input) {
        value = *input;
        *(output++) = (value > 0) ? value : value * scale;
    }
}

void LReluLayer::Backward(Tensor *none) {
    int one_batch_size = info.input_number;
    float *output = output_tensor->weight;
    float *output_delta = output_tensor->delta_weight;
    float *input_delta = input_tensor->delta_weight;
    float chain_grad, scale = kernel[0][0];
    
    for (int b = info.batch_size; b--; ) {
        for (int i = 0; i < info.input_number; ++i) {
            chain_grad = output_delta[i];
            if (output[i] < 0) {
                input_delta[i] += chain_grad * scale;
            }
            else {
                input_delta[i] += chain_grad;
            }
        }
        output += one_batch_size;
        output_delta += one_batch_size;
        input_delta += one_batch_size;
    }
}

SigmoidLayer::SigmoidLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Sigmoid;
    name = (opt.find("name") == opt.end()) ? "sig" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = atoi(opt["input_width"].c_str());
    info.output_height = atoi(opt["input_height"].c_str());
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.output_width * info.output_height * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(Sigmoid);

void SigmoidLayer::Forward(bool train) {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    for (int i = info.input_number * info.batch_size; i--; ) {
        *(output++) = 1.0 / (1.0 + exp(-(*(input++))));
    }
}

void SigmoidLayer::Backward(Tensor *none) {
    float *output = output_tensor->weight;
    float *output_delta = output_tensor->delta_weight;
    float *input_delta = input_tensor->delta_weight;
    float value;
    
    for (int i = 0; i < info.input_number * info.batch_size; ++i) {
        value = *(output++);
        *(input_delta++) = value * (1.0 - value) * *(output_delta++);
    }
}

BatchNormalizationLayer::BatchNormalizationLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::BatchNormalization;
    name = (opt.find("name") == opt.end()) ? "bn" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = atoi(opt["input_width"].c_str());
    info.output_height = atoi(opt["input_height"].c_str());
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.output_width * info.output_height * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    kernel = new Tensor [7];
    kernel[0] = Tensor(1, 1, info.output_dimension, 1); // scale
    kernel[0].extend();
    kernel[1] = Tensor(1, 1, info.output_dimension, 0); // mean
    kernel[1].extend();
    kernel[2] = Tensor(1, 1, info.output_dimension, 0); // variance
    kernel[2].extend();
    kernel[3] = Tensor(1, 1, info.output_dimension, 0); // running mean
    kernel[4] = Tensor(1, 1, info.output_dimension, 0); // running variance
    kernel[5] = Tensor(1, 1, info.input_number * info.batch_size, 0); // x
    kernel[6] = Tensor(1, 1, info.input_number * info.batch_size, 0); // x_norm
    info.kernel_num = 7;
    
    biases = new Tensor(1, 1, info.output_dimension, 0);
    biases->extend();
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(BatchNormalization);

void BatchNormalizationLayer::Forward(bool train) {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    float *bias = biases->weight;
    float *scale = kernel[0].weight;
    float *mean = kernel[1].weight;
    float *variance = kernel[2].weight;
    float *running_mean = kernel[3].weight;
    float *running_variance = kernel[4].weight;
    float *x = kernel[5].weight;
    float *x_norm = kernel[6].weight;
    
    int batch_size = info.batch_size;
    int output_dimension = info.output_dimension;
    int total_size = info.input_number * batch_size;
    int channel_size = info.output_width * info.output_height;
    
    copy_cpu(total_size, input, output);
    copy_cpu(total_size, output, x);
    
    if (train) {
        mean_cpu(output, batch_size, output_dimension, channel_size, mean);
        variance_cpu(output, mean, batch_size, output_dimension, channel_size, variance);
        
        scal_cpu(output_dimension, 0.99, running_mean);
        axpy_cpu(output_dimension, 0.01, mean, running_mean);
        scal_cpu(output_dimension, 0.99, running_variance);
        axpy_cpu(output_dimension, 0.01, variance, running_variance);
        
        normalize(output, mean, variance, batch_size, output_dimension, channel_size);
        copy_cpu(total_size, output, x_norm);
    } else {
        normalize(output, running_mean, running_variance, batch_size, output_dimension, channel_size);
    }
    
    scale_bias(output, scale, batch_size, output_dimension, channel_size);
    add_bias(output, bias, batch_size, output_dimension, channel_size);
}

void BatchNormalizationLayer::Backward(Tensor *none) {
    float *input_delta = input_tensor->delta_weight;
    float *output_delta = output_tensor->delta_weight;
    float *bias_delta = biases->delta_weight;
    float *scale_delta = kernel[0].delta_weight;
    float *mean_delta = kernel[1].delta_weight;
    float *variance_delta = kernel[2].delta_weight;
    
    float *scale = kernel[0].weight;
    float *mean = kernel[1].weight;
    float *variance = kernel[2].weight;
    float *x = kernel[5].weight;
    float *x_norm = kernel[6].weight;
    
    int batch_size = info.batch_size;
    int output_dimension = info.output_dimension;
    int one_batch_size = info.input_number;
    int channel_size = info.output_width * info.output_height;
    
    float chain_grad, scale_value, mean_value;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int d = 0; d < output_dimension; ++d) {
            float &bias_delta_value = bias_delta[d];
            float &scale_delta_value = scale_delta[d];
            float &mean_delta_value = mean_delta[d];
            float &variance_delta_value = variance_delta[d];
            scale_value = scale[d];
            mean_value = mean[d];
            for (int i = 0; i < channel_size; ++i) {
                chain_grad = *output_delta;
                bias_delta_value += chain_grad;
                scale_delta_value += chain_grad * *(x_norm++);
                chain_grad = *(output_delta++) *= scale_value;
                mean_delta_value += chain_grad;
                variance_delta_value += chain_grad * (*(x++) - mean_value);
            }
        }
    }
    
    for (int d = 0; d < output_dimension; ++d) {
        mean_delta[d] *= (-1.0 / sqrt(variance[d] + .00001f));
        variance_delta[d] *= (-0.5 * pow(variance[d] + 0.00001f, (float)(-1.5)));
    }
    
    output_delta = output_tensor->delta_weight;
    x = kernel[5].weight;
    
    float variance_scale;
    float variance_delta_value;
    float mean_delta_value;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int d = 0; d < output_dimension; ++d) {
            mean_value = mean[d];
            mean_delta_value = mean_delta[d] / (channel_size * batch_size);
            variance_scale = 1.0 / (sqrt(variance[d] + .00001f));
            variance_delta_value = variance_delta[d];
            for (int i = 0; i < channel_size; ++i) {
                *output_delta = *output_delta * variance_scale + variance_delta_value * 2.0 * (*(x++) - mean_value) / (channel_size * batch_size) + mean_delta_value;
                ++output_delta;
            }
        }
    }
    
    output_delta = output_tensor->delta_weight;
    copy_cpu(one_batch_size * batch_size, output_delta, input_delta);
    
    //    backward_bias(bias_delta, output_delta, batch_size, output_dimension, channel_size);
    //    backward_scale_cpu(x_norm, output_delta, batch_size, output_dimension, channel_size, scale_delta);
    //
    //    scale_bias(output_delta, scale, batch_size, output_dimension, channel_size);
    //
    //    mean_delta_cpu(output_delta, variance, batch_size, output_dimension, channel_size, mean_delta);
    //    variance_delta_cpu(x, output_delta, mean, variance, batch_size, output_dimension, channel_size, variance_delta);
    //    normalize_delta_cpu(x, mean, variance, mean_delta, variance_delta, batch_size, output_dimension, channel_size, output_delta);
    //    copy_cpu(one_batch_size * batch_size, output_delta, input_delta);
}

void BatchNormalizationLayer::backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates) {
    for(int f = 0; f < n; ++f){
        float sum = 0;
        for(int b = 0; b < batch; ++b){
            for(int i = 0; i < size; ++i){
                int index = i + size * (f + n * b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}

void BatchNormalizationLayer::mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta) {
    for(int i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (int j = 0; j < batch; ++j) {
            for (int k = 0; k < spatial; ++k) {
                int index = j * filters * spatial + i * spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}
void BatchNormalizationLayer::variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta) {
    for(int i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for(int j = 0; j < batch; ++j){
            for(int k = 0; k < spatial; ++k){
                int index = j * filters * spatial + i * spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        variance_delta[i] *= -0.5 * pow(variance[i] + 0.00001, (float)(-1.5));
    }
}
void BatchNormalizationLayer::normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta) {
    for(int j = 0; j < batch; ++j){
        for(int f = 0; f < filters; ++f){
            for(int k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}

UpSampleLayer::UpSampleLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::UpSample;
    name = (opt.find("name") == opt.end()) ? "up" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    int stride = atoi(opt["stride"].c_str());
    info.output_width = stride * info.input_width;
    info.output_height = stride * info.input_height;
    info.output_dimension = info.input_dimension;
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    info.reverse = false;
    
    if (stride < 0) {
        stride = -stride;
        info.reverse = true;
        info.output_width = info.input_width / stride;
        info.output_height = info.input_height / stride;
    }
    info.stride = stride;
    info.batch_size = atoi(opt["batch_size"].c_str());
    info.scale = (opt.find("scale") == opt.end()) ? 1 : atof(opt["scale"].c_str());
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(UpSample);

void UpSampleLayer::Forward(bool train) {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    
    if (info.reverse) {
        upsample(output, info.output_width, info.output_height, info.output_dimension, info.batch_size, info.stride, false, info.scale, input);
    } else {
        upsample(input, info.input_width, info.input_height, info.input_dimension, info.batch_size, info.stride, true, info.scale, output);
    }
}

void UpSampleLayer::Backward(Tensor *none) {
    float *input_delta = input_tensor->delta_weight;
    float *output_delta = output_tensor->delta_weight;
    
    if (info.reverse) {
        upsample(output_delta, info.output_width, info.output_height, info.output_dimension, info.batch_size, info.stride, true, info.scale, input_delta);
    } else {
        upsample(input_delta, info.input_width, info.input_height, info.input_dimension, info.batch_size, info.stride, false, info.scale, output_delta);
    }
}

void UpSampleLayer::upsample(float *in, int w, int h, int c, int batch, int stride, bool forward, float scale, float *out) {
    int input_size = info.input_number;
    int input_channel_size = info.input_width * info.input_height;
    int output_size = info.output_number;
    int output_channel_size = info.output_width * info.output_height;
    
    for(int b = 0; b < batch; ++b){
        for(int k = 0; k < c; ++k){
            for(int j = 0; j < h * stride; ++j){
                for(int i = 0; i < w * stride; ++i){
                    int in_index = b * input_size + k * input_channel_size + (j / stride) * w + i / stride;
                    int out_index = b * output_size + k * output_channel_size + j * w * stride + i;
                    if(forward)
                        out[out_index] = scale * in[in_index];
                    else
                        in[in_index] += scale * out[out_index];
                }
            }
        }
    }
}

ConcatLayer::ConcatLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Concat;
    name = (opt.find("name") == opt.end()) ? "cc" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    
    info.output_width = info.input_width;
    info.output_height = info.input_height;
    info.output_dimension = info.input_dimension;
    
    info.concat_num = atoi(opt["concat_num"].c_str());
    kernel = new Tensor [1];    // Concat tensor structure
    kernel[0] = Tensor(1, 1, info.concat_num + 1, 0);
    kernel[0] = {float(info.input_dimension)};
    info.kernel_num = 1;
    
    for (int i = 1; i <= info.concat_num; ++i) {
        int width_check = atoi(opt[("concat_" + to_string(i) + "_width")].c_str());
        int height_check = atoi(opt[("concat_" + to_string(i) + "_height")].c_str());
        if (width_check != info.input_width || height_check != info.input_height) {
            fprintf(stderr, "[ConcatLayer] Concat shape error!\n");
            fprintf(stderr, "[ConcatLayer] Concat list: %s\n", opt["concat"].c_str());
            fprintf(stderr, "[ConcatLayer] Error index: %d concat(%d * %d) != layer(%d * %d)\n", i - 1, width_check, height_check, info.input_width, info.input_height);
            exit(-100);
        }
        int concat_dimension = kernel[0][i] = atoi(opt[("concat_" + to_string(i) + "_dimension")].c_str());
        info.output_dimension += concat_dimension;
    }
    
    info.splits = (opt.find("splits") == opt.end()) ? 1 : atoi(opt["splits"].c_str());
    info.split_id = (opt.find("split_id") == opt.end()) ? 0 : atoi(opt["split_id"].c_str());
    
    info.output_dimension /= info.splits;
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    
    concat_tensor.resize(info.concat_num + 1);
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(Concat);

Tensor* ConcatLayer::connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace) {
    input_tensor = input_tensor_;
    for (int i = 0; i <= info.concat_num; ++i) {
        concat_tensor[i] = (extra_tensor_[i]);
    }
    return output_tensor;
}

void ConcatLayer::Forward(bool train) {
    float *output = output_tensor->weight;
    
    int channel_size = info.input_width * info.input_height;
    
    for (int b = 0; b < info.batch_size; ++b) {
        for (int i = 0; i <= info.concat_num; ++i) {
            int concat_input_size = channel_size * int(kernel[0][i]);
            int split_concat_input_size = concat_input_size / info.splits;
            float *concat = concat_tensor[i]->weight + b * concat_input_size + split_concat_input_size * info.split_id;
            for (int j = split_concat_input_size; j--; ) {
                *(output++) = *(concat++);
            }
        }
    }
}

void ConcatLayer::Backward(Tensor *none) {
    float *output_delta = output_tensor->delta_weight;
    int check = output_tensor->size;
    
    int channel_size = info.input_width * info.input_height;
    
    for (int b = 0; b < info.batch_size; ++b) {
        for (int i = 0; i <= info.concat_num; ++i) {
            int concat_input_size = channel_size * int(kernel[0][i]);
            int split_concat_input_size = concat_input_size / info.splits;
            float *concat_delta = concat_tensor[i]->delta_weight + b * concat_input_size + split_concat_input_size * info.split_id;
            for (int j = split_concat_input_size; j--; ) {
                *(concat_delta++) += *(output_delta++);
                check--;
            }
        }
    }
}

MishLayer::MishLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Mish;
    name = (opt.find("name") == opt.end()) ? "mish" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    
    info.output_width = info.input_width;
    info.output_height = info.input_height;
    info.output_dimension = info.input_dimension;
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    
    kernel = new Tensor [1];
    kernel[0] = Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);    // Input Storage
    info.kernel_num = 1;
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(Mish);

void MishLayer::Forward(bool train) {
    const float MISH_THRESHOLD = 20;
    float *input = input_tensor->weight;
    float *activation_input = kernel[0].weight;
    float *output = output_tensor->weight;
    
    for (int i = info.input_number * info.batch_size; i--; ) {
        float x_val = *(input++);
        *(activation_input++) = x_val;
        *(output++) = x_val * tanh_activate(softplus_activate(x_val, MISH_THRESHOLD));
    }
}

void MishLayer::Backward(Tensor *none) {
    float *input_delta = input_tensor->delta_weight;
    float *output_delta = output_tensor->delta_weight;
    float *activation_input = kernel[0].weight;
    const float MISH_THRESHOLD = 20.0f;
    
    for (int i = info.output_number * info.batch_size; i--; ) {
        
        // implementation from TensorFlow: https://github.com/tensorflow/addons/commit/093cdfa85d334cbe19a37624c33198f3140109ed
        // implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
        float inp = *(activation_input++);
        float sp = softplus_activate(inp, MISH_THRESHOLD);
        float grad_sp = 1 - exp(-sp);
        float tsp = tanh(sp);
        float grad_tsp = (1 - tsp * tsp) * grad_sp;
        float grad = inp * grad_tsp + tsp;
        *(input_delta++) += grad * *(output_delta++);
    }
}

SwishLayer::SwishLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Swish;
    name = (opt.find("name") == opt.end()) ? "swish" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    
    info.output_width = info.input_width;
    info.output_height = info.input_height;
    info.output_dimension = info.input_dimension;
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    
    kernel = new Tensor [1];
    kernel[0] = Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);    // Input Storage
    info.kernel_num = 1;
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(Swish);

void SwishLayer::Forward(bool train) {
    float *input = input_tensor->weight;
    float *activation_input = kernel[0].weight;
    float *output = output_tensor->weight;
    
    int n = info.input_number * info.batch_size;
//    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; ++i) {
        float x_val = input[i];
        float sigmoid = logistic_activate(x_val);
        activation_input[i] = sigmoid;
        output[i] = x_val * sigmoid;
    }
}

void SwishLayer::Backward(Tensor *none) {
    float *output = output_tensor->weight;
    float *input_delta = input_tensor->delta_weight;
    float *output_delta = output_tensor->delta_weight;
    float *activation_input = kernel[0].weight;
    
    int n = info.output_number * info.batch_size;
//    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float swish = output[i];
        input_delta[i] += output_delta[i] * (swish + activation_input[i] * (1 - swish));
    }
}

DropoutLayer::DropoutLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Dropout;
    name = (opt.find("name") == opt.end()) ? "dropout" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    
    info.output_width = info.input_width;
    info.output_height = info.input_height;
    info.output_dimension = info.input_dimension;
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    
    info.probability = (opt.find("probability") == opt.end()) ? 0.5 : atof(opt["probability"].c_str());
    info.scale = 1.0 / (1.0 - info.probability);
    
    kernel = new Tensor [1];
    kernel[0] = Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);    // Probability storage
    info.kernel_num = 1;
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(Dropout);

void DropoutLayer::Forward(bool train) {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    float *prob = kernel->weight;
    float probability = info.probability;
    float scale = info.scale;
    
    if (!train) {
        copy_cpu(info.input_number * info.batch_size, input, output);
        return;
    }
    
    for(int i = info.input_number * info.batch_size; i--; ++input, ++output){
        float p = Random(0, 1);
        *(prob++) = p;
        if(p < probability)
            *(output) = 0;
        else
            *(output) = *(input) * scale;
    }
}

void DropoutLayer::Backward(Tensor *none) {
    float *input_delta = input_tensor->delta_weight;
    float *output_delta = output_tensor->delta_weight;
    float *prob = kernel->weight;
    float probability = info.probability;
    float scale = info.scale;
    
    for(int i = info.input_number * info.batch_size; i--; ) {
        float p = *(prob++);
        if(p < probability)
            *(input_delta) = 0;
        else
            *(input_delta) = *(output_delta) * scale;
    }
}

AvgPoolingLayer::AvgPoolingLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::AvgPooling;
    name = (opt.find("name") == opt.end()) ? "avgpooling" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    
    info.output_width = 1;
    info.output_height = 1;
    info.output_dimension = info.input_dimension;
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    
    output_tensor = new Tensor(1, 1, info.output_dimension * info.batch_size, 0);
    output_tensor->extend();
}

REGISTER_LAYER_CLASS(AvgPooling);

void AvgPoolingLayer::Forward(bool train) {
    int batch_size = info.batch_size;
    int output_dimension = info.output_dimension;
    int channel_size = info.input_width * info.input_height;
    
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    
    for(int b = 0; b < batch_size; ++b){
        for(int k = 0; k < output_dimension; ++k){
            int out_index = k + b * output_dimension;
            output[out_index] = 0;
            for(int i = 0; i < channel_size; ++i){
                int in_index = i + channel_size * (k + b * output_dimension);
                output[out_index] += input[in_index];
            }
            output[out_index] /= channel_size;
        }
    }
}

void AvgPoolingLayer::Backward(Tensor *none) {
    int batch_size = info.batch_size;
    int output_dimension = info.output_dimension;
    int channel_size = info.input_width * info.input_height;
    
    float *input_delta = input_tensor->delta_weight;
    float *output_delta = output_tensor->delta_weight;
    
    for(int b = 0; b < batch_size; ++b){
        for(int k = 0; k < output_dimension; ++k){
            int out_index = k + b * output_dimension;
            for(int i = 0; i < channel_size; ++i){
                int in_index = i + channel_size * (k + b * output_dimension);
                input_delta[in_index] += output_delta[out_index] / (channel_size);
            }
        }
    }
}

YOLOv3Layer::YOLOv3Layer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Yolov3;
    name = (opt.find("name") == opt.end()) ? "yolov3" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    
    info.output_width = info.input_width;
    info.output_height = info.input_height;
    info.output_dimension = info.input_dimension;
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    
    info.total_anchor_num = atoi(opt["total_anchor_num"].c_str());
    info.anchor_num = atoi(opt["anchor_num"].c_str());
    info.classes = atoi(opt["classes"].c_str());
    info.max_boxes = atoi(opt["max_boxes"].c_str());
    info.net_width = atoi(opt["net_width"].c_str());
    info.net_height = atoi(opt["net_height"].c_str());
    info.ignore_iou_threshold = (opt.find("ignore_iou_threshold") == opt.end()) ? 0.5 : atof(opt["ignore_iou_threshold"].c_str());
    info.truth_iou_threshold = (opt.find("truth_iou_threshold") == opt.end()) ? 1 : atof(opt["truth_iou_threshold"].c_str());
    
    kernel = new Tensor [1];
    kernel[0] = Tensor(1, 1, info.anchor_num, 0); // Mask
    if (opt.find("mask") == opt.end()) {
        for (int i = 0; i < info.anchor_num; ++i)
        kernel[0].weight[i] = i;
    } else {
        stringstream ss;
        ss << opt["mask"];
        int n;
        char c;
        for (int i = 0; i < info.anchor_num; ++i) {
            ss >> n >> c;
            kernel[0].weight[i] = n;
        }
    }
    info.kernel_num = 1;
    
    biases = new Tensor(2, 1, info.total_anchor_num, 0); // Anchor
    stringstream ss;
    ss << opt["anchor"];
    int w, h;
    char c;
    for (int i = 0; i < info.total_anchor_num; ++i) {
        ss >> w >> c >> h;
        biases->weight[i * 2 + 0] = w;
        biases->weight[i * 2 + 1] = h;
    }
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
    output_tensor->extend();
    detection = Tensor(1, 1, 1, 0);
}

REGISTER_LAYER_CLASS(YOLOv3);

Tensor* YOLOv3Layer::connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace) {
    input_tensor = input_tensor_;
    return &detection;
}

void YOLOv3Layer::Forward(bool train) {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    int channel_size = info.output_width * info.output_height;
    
    memcpy(output, input, info.output_number * info.batch_size * sizeof(float));
    
    for (int b = 0; b < info.batch_size; ++b){
        for(int n = 0; n < info.anchor_num; ++n) {
            int index = entry_index(b, n * channel_size, 0);
            activate_array(output + index, 2 * channel_size, LOGISTIC);
            index = entry_index(b, n * channel_size, 4);
            activate_array(output + index, (1 + info.classes) * channel_size, LOGISTIC);
        }
    }
    input_tensor->clearDeltaWeight();
    
    if (!train) {
        vector<Detection> dets = yolo_get_detection_without_correction();
        detection = Tensor(1, (info.classes + 5), (int)dets.size(), 0);
        float *value = detection.weight;
        for (int i = 0; i < (int)dets.size(); ++i) {
            *(value++) = dets[i].bbox.x;
            *(value++) = dets[i].bbox.y;
            *(value++) = dets[i].bbox.w;
            *(value++) = dets[i].bbox.h;
            *(value++) = dets[i].objectness;
            for (int c = 0; c < info.classes; ++c) {
                *(value++) = dets[i].prob[c];
            }
        }
    }
}

void YOLOv3Layer::Backward(Tensor *target) {
    float ignore_iou_threshold = 0.7; // TODO: should be input
    float truth_iou_threshold = 1;
    
    int width = info.input_width;
    int height = info.input_height;
    int net_width = info.net_width;
    int net_height = info.net_height;
    int anchor_num = info.anchor_num;
    int total_anchor_num = info.total_anchor_num;
    int channel_size = width * height;
    int max_boxes = info.max_boxes;
    int target_size = 5 * max_boxes;
    int classes = info.classes;
    
    float *output = output_tensor->weight;
    float *delta = input_tensor->delta_weight;
    float *bias = biases->weight;
    float *mask = kernel[0].weight;
    float *target_ptr = target->weight;
    
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    
    float &loss = info.loss; loss = 0;
    
    #pragma omp parallel for num_threads(4)
    for (int b = 0; b < info.batch_size; ++b) {
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                for (int n = 0; n < anchor_num; ++n) {
                    int box_index = entry_index(b, n * channel_size + j * width + i, 0);
                    Box pred = get_yolo_box(output, bias, mask[n], box_index, i, j, width, height, net_width, net_height, channel_size);
                    float best_iou = 0;
                    int best_t = 0;
                    for(int t = 0; t < max_boxes; ++t){
                        Box truth = float_to_box(target_ptr + t * (4 + 1) + target_size, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_index(b, n * channel_size + j * width + i, 4);
                    avg_anyobj += output[obj_index];
                    delta[obj_index] = output[obj_index];
                    if (best_iou > ignore_iou_threshold) {
                        delta[obj_index] = 0;
                    }
                    if (best_iou > truth_iou_threshold) {
                        delta[obj_index] = output[obj_index] - 1;
                        
                        int cls = target_ptr[best_t*(4 + 1) + b * target_size + 4];
                        int class_index = entry_index(b, n * channel_size + j * width + i, 4 + 1);
                        delta_yolo_class(output, delta, class_index, cls, classes, channel_size, 0);
                        Box truth = float_to_box(target_ptr + best_t * (4 + 1) + b * target_size, 1);
                        delta_yolo_box(truth, output, bias, mask[n], box_index, i, j, width, height, net_width, net_height, delta, (2 - truth.w * truth.h), channel_size);
                    }
                }
            }
        }
        for(int t = 0; t < max_boxes; ++t){
            Box truth = float_to_box(target_ptr + t * (4 + 1) + b * target_size, 1);
            
            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            int i = (truth.x * width);
            int j = (truth.y * height);
            Box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for(int n = 0; n < total_anchor_num; ++n){
                Box pred = {0};
                pred.w = bias[2 * n + 0] / net_width;
                pred.h = bias[2 * n + 1] / net_height;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }
            
            int mask_n = int_index(mask, best_n, anchor_num);
            if(mask_n >= 0){
                int box_index = entry_index(b, mask_n * channel_size + j * width + i, 0);
                float iou = delta_yolo_box(truth, output, bias, best_n, box_index, i, j, width, height, net_width, net_height, delta, (2 - truth.w * truth.h), channel_size);
                
                int obj_index = entry_index(b, mask_n * channel_size + j * width + i, 4);
                avg_obj += output[obj_index];
                delta[obj_index] = output[obj_index] - 1;
                
                int cls = target_ptr[t * (4 + 1) + b * target_size + 4];
                int class_index = entry_index(b, mask_n * channel_size + j * width + i, 4 + 1);
                delta_yolo_class(output, delta, class_index, cls, classes, channel_size, &avg_cat);
                
                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    loss = pow(mag_array(delta, info.output_number * info.batch_size), 2);
    printf("Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(channel_size*anchor_num*info.batch_size), recall/count, recall75/count, count);
}

void YOLOv3Layer::delta_yolo_class(float *output, float *delta, int index, int cls, int classes, int stride, float *avg_cat) {
    if (delta[index]) {
        delta[index + stride * cls] = output[index + stride * cls] - 1;
        if(avg_cat)
            *avg_cat += output[index + stride * cls];
        return;
    }
    for(int n = 0; n < classes; ++n) {
        delta[index + stride * n] = output[index + stride * n] - ((n == cls) ? 1 : 0);
        if(n == cls && avg_cat)
            *avg_cat += output[index + stride * n];
    }
}

float YOLOv3Layer::delta_yolo_box(Box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride) {
    Box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);
    
    float tx = (truth.x * lw - i);
    float ty = (truth.y * lh - j);
    float tw = log(truth.w * w / biases[2 * n + 0]);
    float th = log(truth.h * h / biases[2 * n + 1]);
    
    delta[index + 0 * stride] = -scale * (tx - x[index + 0 * stride]);
    delta[index + 1 * stride] = -scale * (ty - x[index + 1 * stride]);
    delta[index + 2 * stride] = -scale * (tw - x[index + 2 * stride]);
    delta[index + 3 * stride] = -scale * (th - x[index + 3 * stride]);
    return iou;
}

int YOLOv3Layer::entry_index(int batch, int location, int entry) {
    int channel_size = info.output_width * info.output_height;
    int n = location / channel_size;
    int loc = location % channel_size;
    return batch * info.output_number + n * channel_size * (5 + info.classes) + entry * channel_size + loc;
}

vector<Detection> YOLOv3Layer::yolo_get_detection_without_correction() {
    float threshold = 0.24; // TODO: threshold input
    float *feature = output_tensor->weight;
    float *bias = biases->weight;
    float *mask = kernel[0].weight;
    
    int width = info.output_width;
    int height = info.output_height;
    int net_width = info.net_width;
    int net_height = info.net_height;
    int classes = info.classes;
    int channel_size = width * height;
    
    vector<Detection> dets;
    
    for (int i = 0; i < channel_size; ++i){
        int row = i / width;
        int col = i % width;
        for(int n = 0; n < info.anchor_num; ++n){
            int obj_index  = entry_index(0, n * channel_size + i, 4);
            float objectness = feature[obj_index];
            if(objectness <= threshold)
                continue;
            Detection det; det.prob.resize(classes);
            int box_index  = entry_index(0, n * channel_size + i, 0);
            det.bbox = get_yolo_box(feature, bias, mask[n], box_index, col, row, width, height, net_width, net_height, channel_size);
            det.objectness = objectness;
            det.classes = info.classes;
            for(int j = 0; j < info.classes; ++j){
                int class_index = entry_index(0, n * channel_size + i, 4 + 1 + j);
                float prob = objectness * feature[class_index];
                det.prob[j] = (prob > threshold) ? prob : 0;
            }
            dets.push_back(det);
        }
    }
    
    return dets;
}

Box YOLOv3Layer::get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {
    Box b;
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n + 0] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

float YOLOv3Layer::mag_array(float *a, int n) {
    float sum = 0;
    for(int i = 0; i < n; ++i){
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

int YOLOv3Layer::int_index(float *a, int val, int n) {
    for(int i = 0; i < n; ++i) {
        if(a[i] == val)
            return i;
    }
    return -1;
}

YOLOv4Layer::YOLOv4Layer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Yolov4;
    name = (opt.find("name") == opt.end()) ? "yolov4" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    
    info.output_width = info.input_width;
    info.output_height = info.input_height;
    info.output_dimension = info.input_dimension;
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    
    info.total_anchor_num = atoi(opt["total_anchor_num"].c_str());
    info.anchor_num = atoi(opt["anchor_num"].c_str());
    info.classes = atoi(opt["classes"].c_str());
    info.max_boxes = (opt.find("max_boxes") == opt.end()) ? 200 : atoi(opt["max_boxes"].c_str());
    info.net_width = atoi(opt["net_width"].c_str());
    info.net_height = atoi(opt["net_height"].c_str());
    info.ignore_iou_threshold = (opt.find("ignore_iou_threshold") == opt.end()) ? 0.5 : atof(opt["ignore_iou_threshold"].c_str());
    info.truth_iou_threshold = (opt.find("truth_iou_threshold") == opt.end()) ? 1 : atof(opt["truth_iou_threshold"].c_str());
    
    info.scale_x_y = (opt.find("scale_x_y") == opt.end()) ? 1 : atof(opt["scale_x_y"].c_str());
    info.iou_normalizer = (opt.find("iou_normalizer") == opt.end()) ? 0.75 : atof(opt["iou_normalizer"].c_str());
    info.obj_normalizer = (opt.find("obj_normalizer") == opt.end()) ? 1 : atof(opt["obj_normalizer"].c_str());
    info.cls_normalizer = (opt.find("cls_normalizer") == opt.end()) ? 1 : atof(opt["cls_normalizer"].c_str());
    info.delta_normalizer = (opt.find("delta_normalizer") == opt.end()) ? 1 : atof(opt["delta_normalizer"].c_str());
    info.beta_nms = (opt.find("beta_nms") == opt.end()) ? 0.6 : atof(opt["beta_nms"].c_str());
    info.objectness_smooth = (opt.find("objectness_smooth") == opt.end()) ? 0 : atof(opt["objectness_smooth"].c_str());
    info.label_smooth_eps = (opt.find("label_smooth_eps") == opt.end()) ? 0 : atof(opt["label_smooth_eps"].c_str());
    info.max_delta = (opt.find("max_delta") == opt.end()) ? FLT_MAX : atof(opt["max_delta"].c_str());
    info.iou_thresh = (opt.find("iou_thresh") == opt.end()) ? FLT_MAX : atof(opt["iou_thresh"].c_str());
    
    info.yolov4_new_coordinate = (opt.find("new_coordinate") != opt.end());
    info.focal_loss = (opt.find("focal_loss") != opt.end());
    if (opt.find("iou_loss") != opt.end()) {
        string iou_loss = opt["iou_loss"];
        if (iou_loss == "mse") {
            info.iou_loss = MSE;
        } else if (iou_loss == "giou") {
            info.iou_loss = GIOU;
        } else if (iou_loss == "diou") {
            info.iou_loss = DIOU;
        } else if (iou_loss == "ciou") {
            info.iou_loss = CIOU;
        }
    } else {
        info.iou_loss = IOU;
    }
    
    if (opt.find("iou_thresh_kind") != opt.end()) {
        string iou_loss = opt["iou_thresh_kind"];
        if (iou_loss == "iou") {
            info.iou_thresh_kind = IOU;
        } else if (iou_loss == "giou") {
            info.iou_thresh_kind = GIOU;
        } else if (iou_loss == "diou") {
            info.iou_thresh_kind = DIOU;
        } else if (iou_loss == "ciou") {
            info.iou_thresh_kind = CIOU;
        }
    } else {
        info.iou_thresh_kind = IOU;
    }
    
    kernel = new Tensor [4];
    kernel[0] = Tensor(1, 1, info.anchor_num, 0); // Mask
    if (opt.find("mask") == opt.end()) {
        for (int i = 0; i < info.anchor_num; ++i)
        kernel[0].weight[i] = i;
    } else {
        stringstream ss;
        ss << opt["mask"];
        int n;
        char c;
        for (int i = 0; i < info.anchor_num; ++i) {
            ss >> n >> c;
            kernel[0].weight[i] = n;
        }
    }
    kernel[1] = Tensor(1, 1, info.batch_size * info.output_width * info.output_height * info.anchor_num, -1);   // labels
    kernel[2] = Tensor(1, 1, info.batch_size * info.output_width * info.output_height * info.anchor_num, -1);   // class_ids
    kernel[3] = Tensor(1, 1, 1, 0); // classes_multipliers
    info.kernel_num = 4;
    
    biases = new Tensor(2, 1, info.total_anchor_num, 0); // Anchor
    stringstream ss;
    ss << opt["anchor"];
    int w, h;
    char c;
    for (int i = 0; i < info.total_anchor_num; ++i) {
        ss >> w >> c >> h;
        biases->weight[i * 2 + 0] = w;
        biases->weight[i * 2 + 1] = h;
    }
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
    output_tensor->extend();
    detection = Tensor(1, 1, 1, 0);
}

REGISTER_LAYER_CLASS(YOLOv4);

Tensor* YOLOv4Layer::connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace) {
    input_tensor = input_tensor_;
    return &detection;
}

void YOLOv4Layer::Forward(bool train) {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    int channel_size = info.output_width * info.output_height;
    float scale_x_y = info.scale_x_y;
    float scale_x_y_bias = -0.5 * (scale_x_y - 1);
    bool new_coordinate = info.yolov4_new_coordinate;
    
    memcpy(output, input, info.output_number * info.batch_size * sizeof(float));
    
    for (int b = 0; b < info.batch_size; ++b){
        for(int n = 0; n < info.anchor_num; ++n) {
            int bbox_index = entry_index(b, n * channel_size, 0);
            if (new_coordinate) {
                
            } else {
                activate_array(output + bbox_index, 2 * channel_size, LOGISTIC);
                int obj_index = entry_index(b, n * channel_size, 4);
                activate_array(output + obj_index, (1 + info.classes) * channel_size, LOGISTIC);
            }
            scal_add_cpu(2 * channel_size, scale_x_y, scale_x_y_bias, output + bbox_index);
        }
    }
    input_tensor->clearDeltaWeight();
    
    if (!train) {
        vector<Detection> dets = yolo_get_detection_without_correction();
        detection = Tensor(1, (info.classes + 5), (int)dets.size(), 0);
        float *value = detection.weight;
        for (int i = 0; i < (int)dets.size(); ++i) {
            *(value++) = dets[i].bbox.x;
            *(value++) = dets[i].bbox.y;
            *(value++) = dets[i].bbox.w;
            *(value++) = dets[i].bbox.h;
            *(value++) = dets[i].objectness;
            for (int c = 0; c < info.classes; ++c) {
                *(value++) = dets[i].prob[c];
            }
        }
    }
}

void YOLOv4Layer::Backward(Tensor *target) {
    int width = info.input_width;
    int height = info.input_height;
    int net_width = info.net_width;
    int net_height = info.net_height;
    int anchor_num = info.anchor_num;
    int total_anchor_num = info.total_anchor_num;
    int channel_size = width * height;
    int max_boxes = info.max_boxes;
    int truth_size = 4 + 2; // x y w h id track_id
    int truths = truth_size * max_boxes;
    int classes = info.classes;
    int batch_size = info.batch_size;
    bool new_coordinate = info.yolov4_new_coordinate;
    float ignore_thresh = info.ignore_iou_threshold;
    float truth_thresh = info.truth_iou_threshold;
    float obj_normalizer = info.obj_normalizer;
    float iou_normalizer = info.iou_normalizer;
    float cls_normalizer = info.cls_normalizer;
    float objectness_smooth = info.objectness_smooth;
    float label_smooth_eps = info.label_smooth_eps;
    float max_delta = info.max_delta;
    bool focal_loss = info.focal_loss;
    float iou_thresh = info.iou_thresh;
    
    IOU_KIND iou_loss = (IOU_KIND)info.iou_loss;
    IOU_KIND iou_thresh_kind = (IOU_KIND)info.iou_thresh_kind;
    
    float *output = output_tensor->weight;
    float *delta = input_tensor->delta_weight;
    float *bias = biases->weight;
    float *mask = kernel[0].weight;
    float *target_ptr = target->weight;
//    float *classes_multipliers = kernel[4].weight;
    float *classes_multipliers = nullptr;
    
    kernel[1] = -1;
    kernel[2] = -1;
    float *labels = kernel[1].weight;
    float *class_ids = kernel[2].weight;
    int rewritten_bbox = 0;
    
    float tot_iou = 0;
    float tot_giou = 0;
    float tot_diou = 0;
    float tot_ciou = 0;
    float tot_iou_loss = 0;
    float tot_giou_loss = 0;
    float tot_diou_loss = 0;
    float tot_ciou_loss = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    
    float &loss = info.loss; loss = 0;
    
    #pragma omp parallel for num_threads(4)
    for (int b = 0; b < batch_size; ++b) {
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                for (int n = 0; n < anchor_num; ++n) {
                    int class_index = entry_index(b, n * channel_size + j * width + i, 4 + 1);
                    int obj_index = entry_index(b, n * channel_size + j * width + i, 4);
                    int box_index = entry_index(b, n * channel_size + j * width + i, 0);
                    Box pred = get_yolo_box(output, bias, mask[n], box_index, i, j, width, height, net_width, net_height, channel_size, new_coordinate);
                    float best_match_iou = 0;
                    int best_match_t = 0;
                    float best_iou = 0;
                    int best_t = 0;
                    for (int t = 0; t < max_boxes; ++t) {
                        Box truth = float_to_box(target_ptr + t * truth_size + b * truths, 1);
                        if (!truth.x) break;  // continue;
                        int class_id = target_ptr[t * truth_size + b * truths + 4];
                        float objectness = output[obj_index];
                        if (isnan(objectness) || isinf(objectness)) output[obj_index] = 0;
                        int class_id_match = compare_yolo_class(output, classes, class_index, channel_size, objectness, class_id, 0.25f);
                        
                        float iou = box_iou(pred, truth);
                        if (iou > best_match_iou && class_id_match == 1) {
                            best_match_iou = iou;
                            best_match_t = t;
                        }
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    
                    avg_anyobj += output[obj_index];
                    delta[obj_index] = -(obj_normalizer * (0 - output[obj_index]));
                    if (best_match_iou > ignore_thresh) {
                        if (objectness_smooth) {
                            const float delta_obj = obj_normalizer * (best_match_iou - output[obj_index]);
                            if (delta_obj < delta[obj_index])
                                delta[obj_index] = -delta_obj;
                            
                        }
                        else delta[obj_index] = 0;
                    }
                    if (best_iou > truth_thresh) {
                        const float iou_multiplier = best_iou * best_iou;// (best_iou - l.truth_thresh) / (1.0 - l.truth_thresh);
                        if (objectness_smooth)
                            delta[obj_index] = -obj_normalizer * (iou_multiplier - output[obj_index]);
                        else
                            delta[obj_index] = -obj_normalizer * (1 - output[obj_index]);
                        
                        int class_id = target_ptr[best_t * truth_size + b * truths + 4];
                        delta_yolo_class(output, delta, class_index, class_id, classes, channel_size, 0, focal_loss, label_smooth_eps, classes_multipliers, cls_normalizer);
                        float class_multiplier = (classes_multipliers) ? classes_multipliers[class_id] : 1.0f;
                        if (objectness_smooth)
                            delta[class_index + channel_size * class_id] = -class_multiplier * (iou_multiplier - output[class_index + channel_size * class_id]);
                        Box truth = float_to_box(target_ptr + best_t * truth_size + b * truths, 1);
                        delta_yolo_box(truth, output, bias, mask[n], box_index, i, j, width, height, net_width, net_height, delta, (2 - truth.w * truth.h), channel_size, iou_normalizer * class_multiplier, iou_loss, 1, max_delta, &rewritten_bbox, new_coordinate);
//                        (*state.net.total_bbox)++;
                    }
                }
            }
        }
        for (int t = 0; t < max_boxes; ++t) {
            Box truth = float_to_box(target_ptr + t * truth_size + b * truths, 1);
            if (!truth.x) break;  // continue;
            int class_id = target_ptr[t * truth_size + b * truths + 4];
            if (class_id >= classes || class_id < 0) continue; // if label contains class_id more than number of classes in the cfg-file and class_id check garbage value
            
            float best_iou = 0;
            int best_n = 0;
            int i = (truth.x * width);
            int j = (truth.y * height);
            Box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for (int n = 0; n < total_anchor_num; ++n) {
                Box pred = { 0 };
                pred.w = bias[2 * n + 0] / net_width;
                pred.h = bias[2 * n + 1] / net_height;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_n = n;
                }
            }
            
            int mask_n = int_index(mask, best_n, anchor_num);
            if (mask_n >= 0) {
                int class_id = target_ptr[t * truth_size + b * truths + 4];
                
                int box_index = entry_index(b, mask_n * channel_size + j * width + i, 0);
                float class_multiplier = (classes_multipliers) ? classes_multipliers[class_id] : 1.0f;
                Ious all_ious = delta_yolo_box(truth, output, bias, best_n, box_index, i, j, width, height, net_width, net_height, delta, (2 - truth.w * truth.h), channel_size, iou_normalizer * class_multiplier, iou_loss, 1, max_delta, &rewritten_bbox, new_coordinate);
//                (*state.net.total_bbox)++;
                
                int truth_in_index = t * truth_size + b * truths + 5;
                int track_id = target_ptr[truth_in_index];
                int truth_out_index = b * anchor_num * channel_size + mask_n * channel_size + j * width + i;
                labels[truth_out_index] = track_id;
                class_ids[truth_out_index] = class_id;
                
                // range is 0 <= 1
                tot_iou += all_ious.iou;
                tot_iou_loss += 1 - all_ious.iou;
                // range is -1 <= giou <= 1
                tot_giou += all_ious.giou;
                tot_giou_loss += 1 - all_ious.giou;
                
                tot_diou += all_ious.diou;
                tot_diou_loss += 1 - all_ious.diou;
                
                tot_ciou += all_ious.ciou;
                tot_ciou_loss += 1 - all_ious.ciou;
                
                int obj_index = entry_index(b, mask_n * channel_size + j * width + i, 4);
                avg_obj += output[obj_index];
                if (objectness_smooth) {
                    float delta_obj = class_multiplier * obj_normalizer * (1 - output[obj_index]);
                    if (delta[obj_index] == 0)
                        delta[obj_index] = -delta_obj;
                }
                else
                    delta[obj_index] = -class_multiplier * obj_normalizer * (1 - output[obj_index]);
                
                int class_index = entry_index(b, mask_n * channel_size + j * width + i, 4 + 1);
                delta_yolo_class(output, delta, class_index, class_id, classes, channel_size, &avg_cat, focal_loss, label_smooth_eps, classes_multipliers, cls_normalizer);
                
                ++count;
                ++class_count;
                if (all_ious.iou > .5) recall += 1;
                if (all_ious.iou > .75) recall75 += 1;
            }
            
            // iou_thresh
            for (int n = 0; n < total_anchor_num; ++n) {
                int mask_n = int_index(mask, n, anchor_num);
                if (mask_n >= 0 && n != best_n && iou_thresh < 1.0f) {
                    Box pred = { 0 };
                    pred.w = bias[2 * n + 0] / net_width;
                    pred.h = bias[2 * n + 1] / net_height;
                    float iou = box_iou_kind(pred, truth_shift, iou_thresh_kind); // IOU, GIOU, MSE, DIOU, CIOU
                    // iou, n
                    
                    if (iou > iou_thresh) {
                        int class_id = target_ptr[t * truth_size + b * truths + 4];
                        
                        int box_index = entry_index(b, mask_n * channel_size + j * width + i, 0);
                        float class_multiplier = (classes_multipliers) ? classes_multipliers[class_id] : 1.0f;
                        Ious all_ious = delta_yolo_box(truth, output, bias, n, box_index, i, j, width, height, net_width, net_height, delta, (2 - truth.w * truth.h), channel_size, iou_normalizer * class_multiplier, iou_loss, 1, max_delta, &rewritten_bbox, new_coordinate);
//                        (*state.net.total_bbox)++;
                        
                        // range is 0 <= 1
                        tot_iou += all_ious.iou;
                        tot_iou_loss += 1 - all_ious.iou;
                        // range is -1 <= giou <= 1
                        tot_giou += all_ious.giou;
                        tot_giou_loss += 1 - all_ious.giou;
                        
                        tot_diou += all_ious.diou;
                        tot_diou_loss += 1 - all_ious.diou;
                        
                        tot_ciou += all_ious.ciou;
                        tot_ciou_loss += 1 - all_ious.ciou;
                        
                        int obj_index = entry_index(b, mask_n * channel_size + j * width + i, 4);
                        avg_obj += output[obj_index];
                        if (objectness_smooth) {
                            float delta_obj = class_multiplier * obj_normalizer * (1 - output[obj_index]);
                            if (delta[obj_index] == 0)
                                delta[obj_index] = -delta_obj;
                        }
                        else
                            delta[obj_index] = -class_multiplier * obj_normalizer * (1 - output[obj_index]);
                        
                        int class_index = entry_index(b, mask_n * channel_size + j * width + i, 4 + 1);
                        delta_yolo_class(output, delta, class_index, class_id, classes, channel_size, &avg_cat, focal_loss, label_smooth_eps, classes_multipliers, cls_normalizer);
                        
                        ++count;
                        ++class_count;
                        if (all_ious.iou > .5) recall += 1;
                        if (all_ious.iou > .75) recall75 += 1;
                    }
                }
            }
        }
        
        if (iou_thresh < 1.0f) {
            // averages the deltas obtained by the function: delta_yolo_box()_accumulate
            for (int j = 0; j < height; ++j) {
                for (int i = 0; i < width; ++i) {
                    for (int n = 0; n < anchor_num; ++n) {
                        int obj_index = entry_index(b, n * channel_size + j * width + i, 4);
                        int box_index = entry_index(b, n * channel_size + j * width + i, 0);
                        int class_index = entry_index(b, n * channel_size + j * width + i, 4 + 1);
                        
                        if (delta[obj_index] != 0)
                            averages_yolo_deltas(class_index, box_index, channel_size, classes, delta);
                    }
                }
            }
        }
        
    }
    
    if (count == 0) count = 1;
    if (class_count == 0) class_count = 1;
    
    loss = pow(mag_array(delta, info.output_number * batch_size), 2);

    loss /= batch_size;

//    fprintf(stderr, "v3 (%s loss, Normalizer: (iou: %.2f, obj: %.2f, cls: %.2f) Avg (IOU: %f), count: %d, total_loss = %f \n", (iou_loss == MSE ? "mse" : (iou_loss == GIOU ? "giou" : "iou")), iou_normalizer, obj_normalizer, cls_normalizer, tot_iou / count, count, loss);
    
    fprintf(stderr, "v3 (%s loss, Normalizer: (iou: %.2f, obj: %.2f, cls: %.2f) Avg (IOU: %f, GIOU: %f), Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, count: %d, total_loss = %f \n", (iou_loss == MSE ? "mse" : (iou_loss == GIOU ? "giou" : "iou")), iou_normalizer, obj_normalizer, cls_normalizer, tot_iou / count, tot_giou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (channel_size * anchor_num * batch_size), recall / count, recall75 / count, count, loss);
}

void YOLOv4Layer::averages_yolo_deltas(int class_index, int box_index, int stride, int classes, float *delta) {

    int classes_in_one_box = 0;
    int c;
    for (c = 0; c < classes; ++c) {
        if (delta[class_index + stride * c] < 0)
            classes_in_one_box++;
    }

    if (classes_in_one_box > 0) {
        delta[box_index + 0 * stride] /= classes_in_one_box;
        delta[box_index + 1 * stride] /= classes_in_one_box;
        delta[box_index + 2 * stride] /= classes_in_one_box;
        delta[box_index + 3 * stride] /= classes_in_one_box;
    }
}

void YOLOv4Layer::delta_yolo_class(float *output, float *delta, int index, int class_id, int classes, int stride, float *avg_cat, int focal_loss, float label_smooth_eps, float *classes_multipliers, float cls_normalizer) {
    if (delta[index + stride * class_id]) {
        float y_true = 1;
        if (label_smooth_eps)
            y_true = y_true *  (1 - label_smooth_eps) + 0.5 * label_smooth_eps;
        float result_delta = y_true - output[index + stride * class_id];
        if(!isnan(result_delta) && !isinf(result_delta))
            delta[index + stride * class_id] = -result_delta;
        //delta[index + stride*class_id] = 1 - output[index + stride*class_id];

        if (classes_multipliers)
            delta[index + stride * class_id] *= classes_multipliers[class_id];
        if (avg_cat)
            *avg_cat += output[index + stride * class_id];
        return;
    }
    // Focal loss
    if (focal_loss) {
        // Focal Loss
        float alpha = 0.5;    // 0.25 or 0.5
        //float gamma = 2;    // hardcoded in many places of the grad-formula

        int ti = index + stride * class_id;
        float pt = output[ti] + 0.000000000000001F;
        float grad = -(1 - pt) * (2 * pt * logf(pt) + pt - 1);

        for (int n = 0; n < classes; ++n) {
            delta[index + stride * n] = -((((n == class_id) ? 1 : 0) - output[index + stride*n]));

            delta[index + stride * n] *= alpha * grad;

            if (n == class_id && avg_cat)
                *avg_cat += output[index + stride * n];
        }
    } else {
        // default
        for (int n = 0; n < classes; ++n) {
            float y_true = ((n == class_id) ? 1 : 0);
            if (label_smooth_eps)
                y_true = y_true *  (1 - label_smooth_eps) + 0.5 * label_smooth_eps;
            float result_delta = y_true - output[index + stride * n];
            if (!isnan(result_delta) && !isinf(result_delta))
                delta[index + stride * n] = -result_delta;

            if (classes_multipliers && n == class_id)
                delta[index + stride * class_id] *= classes_multipliers[class_id] * cls_normalizer;
            if (n == class_id && avg_cat)
                *avg_cat += output[index + stride*n];
        }
    }
}

Ious YOLOv4Layer::delta_yolo_box(Box &truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride, float iou_normalizer, IOU_KIND iou_loss, int accumulate, float max_delta, int *rewritten_bbox, int new_coords) {
    if (delta[index + 0 * stride] || delta[index + 1 * stride] || delta[index + 2 * stride] || delta[index + 3 * stride]) {
        (*rewritten_bbox)++;
    }

    Ious all_ious = {0};
    // i - step in layer width
    // j - step in layer height
    //  Returns a box in absolute coordinates
    Box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride, new_coords);
    all_ious.iou = box_iou(pred, truth);
    all_ious.giou = box_giou(pred, truth);
    all_ious.diou = box_diou(pred, truth);
    all_ious.ciou = box_ciou(pred, truth);
    // avoid nan in dx_box_iou
    if (pred.w == 0) { pred.w = 1.0; }
    if (pred.h == 0) { pred.h = 1.0; }
    if (iou_loss == MSE)    // old loss
    {
        float tx = (truth.x * lw - i);
        float ty = (truth.y * lh - j);
        float tw = log(truth.w * w / biases[2 * n + 0]);
        float th = log(truth.h * h / biases[2 * n + 1]);

        if (new_coords) {
            tw = sqrt(truth.w*w / (4 * biases[2 * n]));
            th = sqrt(truth.h*h / (4 * biases[2 * n + 1]));
        }

        // accumulate delta
        delta[index + 0 * stride] += -scale * (tx - x[index + 0 * stride]) * iou_normalizer;
        delta[index + 1 * stride] += -scale * (ty - x[index + 1 * stride]) * iou_normalizer;
        delta[index + 2 * stride] += -scale * (tw - x[index + 2 * stride]) * iou_normalizer;
        delta[index + 3 * stride] += -scale * (th - x[index + 3 * stride]) * iou_normalizer;
    }
    else {
        all_ious.dx_iou = dx_box_iou(pred, truth, iou_loss);

        // jacobian^t (transpose)
        //float dx = (all_ious.dx_iou.dl + all_ious.dx_iou.dr);
        //float dy = (all_ious.dx_iou.dt + all_ious.dx_iou.db);
        //float dw = ((-0.5 * all_ious.dx_iou.dl) + (0.5 * all_ious.dx_iou.dr));
        //float dh = ((-0.5 * all_ious.dx_iou.dt) + (0.5 * all_ious.dx_iou.db));

        // jacobian^t (transpose)
        float dx = all_ious.dx_iou.dt;
        float dy = all_ious.dx_iou.db;
        float dw = all_ious.dx_iou.dl;
        float dh = all_ious.dx_iou.dr;


        // predict exponential, apply gradient of e^delta_t ONLY for w,h
        if (new_coords) {
            //dw *= 8 * x[index + 2 * stride];
            //dh *= 8 * x[index + 3 * stride];
            //dw *= 8 * x[index + 2 * stride] * biases[2 * n] / w;
            //dh *= 8 * x[index + 3 * stride] * biases[2 * n + 1] / h;

            //float grad_w = 8 * exp(-x[index + 2 * stride]) / pow(exp(-x[index + 2 * stride]) + 1, 3);
            //float grad_h = 8 * exp(-x[index + 3 * stride]) / pow(exp(-x[index + 3 * stride]) + 1, 3);
            //dw *= grad_w;
            //dh *= grad_h;
        }
        else {
            dw *= exp(x[index + 2 * stride]);
            dh *= exp(x[index + 3 * stride]);
        }

        // normalize iou weight
        dx *= iou_normalizer;
        dy *= iou_normalizer;
        dw *= iou_normalizer;
        dh *= iou_normalizer;

        dx = fix_nan_inf(dx);
        dy = fix_nan_inf(dy);
        dw = fix_nan_inf(dw);
        dh = fix_nan_inf(dh);

        if (max_delta != FLT_MAX) {
            dx = clip_value(dx, max_delta);
            dy = clip_value(dy, max_delta);
            dw = clip_value(dw, max_delta);
            dh = clip_value(dh, max_delta);
        }

        if (!accumulate) {
            delta[index + 0 * stride] = 0;
            delta[index + 1 * stride] = 0;
            delta[index + 2 * stride] = 0;
            delta[index + 3 * stride] = 0;
        }

        // accumulate delta
        delta[index + 0 * stride] += -dx;
        delta[index + 1 * stride] += -dy;
        delta[index + 2 * stride] += -dw;
        delta[index + 3 * stride] += -dh;
    }

    return all_ious;
}

int YOLOv4Layer::entry_index(int batch, int location, int entry) {
    int channel_size = info.output_width * info.output_height;
    int n = location / channel_size;
    int loc = location % channel_size;
    return batch * info.output_number + n * channel_size * (5 + info.classes) + entry * channel_size + loc;
}

int YOLOv4Layer::compare_yolo_class(float *output, int classes, int class_index, int stride, float objectness, int class_id, float conf_thresh) {
    for (int j = 0; j < classes; ++j) {
        float prob = output[class_index + stride * j];
        if (prob > conf_thresh) {
            return 1;
        }
    }
    return 0;
}

vector<Detection> YOLOv4Layer::yolo_get_detection_without_correction() {
    float threshold = 0.24; // TODO: threshold input
    float *feature = output_tensor->weight;
    float *bias = biases->weight;
    float *mask = kernel[0].weight;
    
    int width = info.output_width;
    int height = info.output_height;
    int net_width = info.net_width;
    int net_height = info.net_height;
    int classes = info.classes;
    int channel_size = width * height;
    
    vector<Detection> dets;
    
    for (int i = 0; i < channel_size; ++i){
        int row = i / width;
        int col = i % width;
        for(int n = 0; n < info.anchor_num; ++n){
            int obj_index  = entry_index(0, n * channel_size + i, 4);
            float objectness = feature[obj_index];
            if(objectness <= threshold)
                continue;
            Detection det; det.prob.resize(classes);
            int box_index  = entry_index(0, n * channel_size + i, 0);
            det.bbox = get_yolo_box(feature, bias, mask[n], box_index, col, row, width, height, net_width, net_height, channel_size, info.yolov4_new_coordinate);
            det.objectness = objectness;
            det.classes = info.classes;
            for(int j = 0; j < info.classes; ++j){
                int class_index = entry_index(0, n * channel_size + i, 4 + 1 + j);
                float prob = objectness * feature[class_index];
                det.prob[j] = (prob > threshold) ? prob : 0;
            }
            dets.push_back(det);
        }
    }
    
    return dets;
}

Box YOLOv4Layer::get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride, bool new_coordinate) {
    Box b;
    if (new_coordinate) {
        b.x = (i + x[index + 0 * stride]) / lw;
        b.y = (j + x[index + 1 * stride]) / lh;
        b.w = x[index + 2 * stride] * x[index + 2 * stride] * 4 * biases[2 * n] / w;
        b.h = x[index + 3 * stride] * x[index + 3 * stride] * 4 * biases[2 * n + 1] / h;
    } else {
        b.x = (i + x[index + 0 * stride]) / lw;
        b.y = (j + x[index + 1 * stride]) / lh;
        b.w = exp(x[index + 2 * stride]) * biases[2 * n + 0] / w;
        b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    }
    return b;
}

float YOLOv4Layer::mag_array(float *a, int n) {
    float sum = 0;
    for(int i = 0; i < n; ++i){
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

int YOLOv4Layer::int_index(float *a, int val, int n) {
    for(int i = 0; i < n; ++i) {
        if(a[i] == val)
            return i;
    }
    return -1;
}

void add_bias(float *output, float *biases, int batch, int n, int size) {
    for(int b = 0; b < batch; ++b){
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < size; ++j){
                output[(b * n + i) * size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size) {
    for(int b = 0; b < batch; ++b){
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < size; ++j){
                output[(b * n + i) * size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size) {
    for(int b = 0; b < batch; ++b){
        for(int i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta + size * (i + b * n), size);
        }
    }
}
