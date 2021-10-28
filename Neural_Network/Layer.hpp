//
//  Layer.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#ifndef Layer_hpp
#define Layer_hpp

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <map>
#include <fstream>
#include <cstring>
#include <unordered_map>
#include <cassert>
#include <sstream>
#include <cmath>

#include "Tensor.hpp"

#ifndef OMP_THREADS
#define OMP_THREADS 4
#endif

using namespace std;

typedef vector<Tensor> vtensor;
typedef vector<Tensor*> vtensorptr;
typedef unordered_map<string, string> LayerOption;

struct Train_Args {
    Train_Args() {valid = false;}
    Train_Args(Tensor *kernel_, Tensor *biases_, int kernel_size_, int kernel_list_size_, int biases_size_, vfloat ln_decay_list_) : valid(true), kernel(kernel_), biases(biases_), kernel_size(kernel_size_), kernel_list_size(kernel_list_size_), biases_size(biases_size_), ln_decay_list(ln_decay_list_) {}
    bool valid;
    Tensor *kernel;
    Tensor *biases;
    int kernel_size;
    int kernel_list_size;
    int biases_size;
    vfloat ln_decay_list;
};

enum LayerType {
    Input,
    Data,
    Pooling,
    Convolution,
    AvgPooling,
    UpSample,
    Dropout,
    FullyConnected,
    Sigmoid,
    Tanh,
    Relu,
    PRelu,
    LRelu,
    Mish,
    Swish,
    Elu,
    BatchNormalization,
    Concat,
    Eltwise,
    ShortCut,
    ScaleChannel,
    Softmax,
    EuclideanLoss,
    Yolov3,
    Yolov4,
    Error
};

// Data layer
class InputLayer;
class DataLayer;

// Vision layers
class ConvolutionLayer;
class PoolingLayer;
class AvgPoolingLayer;
class UpSampleLayer;

// Common layers
class DropoutLayer;
class FullyConnectedLayer;

// Activations layers
class SigmoidLayer;
class TanhLayer;
class ReluLayer;
class PReluLayer;
class LReluLayer;
class MishLayer;
class SwishLayer;
class EluLayer;

// Normalization layer
class BatchNormalizationLayer;

// Utility layer
class ConcatLayer;
class EltwiseLayer;
class ShortCutLayer;
class ScaleChannelLayer;

// Loss layers
class SoftmaxLayer;
class EuclideanLossLayer;

// Special layers
class YOLOv3Layer;
class YOLOv4Layer;

#define opt_find(opt, type) \
    (opt.find(type) != opt.end())
#define opt_get_int(opt, type) \
    atoi(opt[type].c_str());
#define opt_get_float(opt, type) \
    atof(opt[type].c_str());
#define opt_find_int(opt, type, default) \
    (opt.find(type) == opt.end()) ? default : opt_get_int(opt, type)
#define opt_find_float(opt, type, default) \
    (opt.find(type) == opt.end()) ? default : opt_get_float(opt, type)
#define opt_find_string(opt, type, default) \
    (opt.find(type) == opt.end()) ? default : opt[type];

// Base layer
class BaseLayer {
public:
    ~BaseLayer();
    BaseLayer();
    BaseLayer(const BaseLayer &L);
    BaseLayer(BaseLayer &&L);
    BaseLayer& operator=(const BaseLayer &L);
    virtual void Forward(Tensor *input_tensor = nullptr) {}
    virtual void Forward(bool train = false) {}
    virtual void Backward(Tensor *target = nullptr) {}
    virtual Tensor* connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace = nullptr);
    virtual inline const char* Type() const {return "Unknown"; }
    void applyKernel(int num);
    void shape();
    void show_detail();
    long getOperations();
    int getWorkspaceSize();
    int getParameter(int type);
    void ClearGrad();
    bool save_raw(FILE *f);
    bool load_raw(FILE *f);
    bool to_prototxt(FILE *f, int refine_id, vector<LayerOption> &refine_struct, unordered_map<string, int> &id_table);
    Train_Args getTrainArgs();
    float getLoss() {return info.loss;}
//protected:
    LayerType type;
    string name;
    string input_name;
    struct Info {
        int input_number;
        int output_number;
        int output_width;
        int output_height;
        int output_dimension;
        int input_width;
        int input_height;
        int input_dimension;
        int concat_num;
        int splits;
        int split_id;
        int shortcut_width;
        int shortcut_height;
        int shortcut_dimension;
        int kernel_width;
        int kernel_height;
        int stride_x;
        int stride_y;
        int stride;
        int padding;
        int groups;
        int dilation;
        int nweights;
        int workspace_size;
        int batch_size;
        int eltwise_num;
        int eltwise_op;
        int kernel_num;
        int total_anchor_num;
        int anchor_num;
        int classes;
        int max_boxes;
        int net_width;
        int net_height;
        float ignore_iou_threshold;
        float truth_iou_threshold;
        float scale_x_y;
        float iou_normalizer;
        float obj_normalizer;
        float cls_normalizer;
        float delta_normalizer;
        float objectness_smooth;
        float label_smooth_eps;
        float max_delta;
        bool focal_loss;
        int iou_loss;
        int iou_thresh_kind;
        int nms_kind;
        float beta_nms;
        float iou_thresh;
        bool yolov4_new_coordinate;
        float probability;
        float scale;
        bool scale_wh;
        bool reverse;
        bool batchnorm;
        float loss;
    } info;
    LayerOption opt;
    Tensor* input_tensor;
    Tensor* output_tensor;
    Tensor* kernel;
    Tensor* biases;
};

class LayerRegistry {
public:
    typedef BaseLayer* (*Creator)(const LayerOption&);
    typedef map<string, Creator> CreatorRegistry;

    static CreatorRegistry &Registry() {
        static auto *g_registry_ = new CreatorRegistry();
        return *g_registry_;
    }

    static void AddCreator(const string &type, Creator creator) {
        CreatorRegistry &registry = Registry();
        if (registry.count(type) == 1) {
            cout << "BaseLayer type " << type << " already registered."<<endl;
        }
        registry[type] = creator;
    }

    static BaseLayer* CreateLayer(LayerOption& opt) {
        string type = opt["type"];
        CreatorRegistry &registry = Registry();
        if (registry.count(type) == 0) {
            cout << "Unknown layer type: " << type << " (known layer types: " << TypeListString() << ")" << endl;
            exit(100);
        }
        return registry[type](opt);
    }

    static vector<string> TypeList() {
        CreatorRegistry &registry = Registry();
        vector<string> types;
        for (typename CreatorRegistry::iterator iter = registry.begin();
             iter != registry.end(); ++iter) {
            types.push_back(iter->first);
        }
        return types;
    }

private:
    LayerRegistry() {}

    static string TypeListString() {
        vector<string> types = TypeList();
        string types_str;
        for (auto iter = types.begin();
             iter != types.end(); ++iter) {
            if (iter != types.begin()) {
                types_str += ", ";
            }
            types_str += *iter;
        }
        return types_str;
    }
};

class LayerRegister {
public:
    LayerRegister(const string &type, BaseLayer* (*creator)(const LayerOption &)) {
        LayerRegistry::AddCreator(type, creator);
    }
};

#define REGISTER_LAYER_CREATOR(type, creator)   \
    static LayerRegister g_creator_##type(#type, creator)

#define REGISTER_LAYER_CLASS(type)  \
    BaseLayer* Creator_##type##Layer(const LayerOption& param) {    \
        return new type##Layer(param);  \
    }   \
    REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

enum STORE_TYPE {
    OPTION,
    REQUIRED
};

struct ParameterData {
    ParameterData() {}
    ParameterData(STORE_TYPE store_, string type_, string name_, string data_) : store(store_), name(name_), type(type_), data(data_) {}
    STORE_TYPE store;
    string name;
    string type;
    string data;
};

class Parameter {
public:
    Parameter() {}
    Parameter(string name_) : name(name_) {}
    string getName() {return name;}
    void addParameter(ParameterData param) {
        parameter.push_back(param);
    }
    vector<ParameterData> getParameter() {return parameter;}
    bool check(string param) {
        for (int i = 0; i < parameter.size(); ++i) {
            if (parameter[i].name == param)
                return true;
        }
        return false;
    }
    ParameterData get(string param) {
        for (int i = 0; i < parameter.size(); ++i) {
            if (parameter[i].name == param)
                return parameter[i];
        }
        return ParameterData();
    }
private:
    string name;
    vector<ParameterData> parameter;
};

class LayerParameter {
public:
    typedef map<string, Parameter> ParameterPool;
    
    static ParameterPool& Pool() {
        static auto *pool = new ParameterPool();
        return *pool;
    }
    
    static void AddParameter(Parameter param) {
        ParameterPool &pool = Pool();
        string name = param.getName();
        if (pool.count(name) == 1) {
            cout << "Layer type: [" << name << "] has been registered." << endl;
            return;
        }
        pool[name] = param;
    }
    
    static Parameter getParameter(string type) {
        ParameterPool &pool = Pool();
        if (pool.count(type) == 0) {
            return Parameter();
        }
        return pool[type];
    }
private:
    LayerParameter() {}
};

class ParameterParser {
public:
    ParameterParser();
    ParameterParser(const char *param_filename);
private:
    void parse();
    vector<string> param_list;
};

#define CHECK_IF_QUIT(str, cmp)  \
if (str == cmp) fprintf(stderr, "[ParameterParser] Unexpected \"%s\"!\n", str.c_str());

#define CHECK_IFNOT_QUIT(str, cmp)  \
if (str != cmp) fprintf(stderr, "[ParameterParser] Expect '" cmp "'!\n");

// Input layer
class InputLayer : public BaseLayer {
public:
    InputLayer(LayerOption opt_);
    void Forward(Tensor *input_tensor_);
    inline const char* Type() const {return "Input"; }
};

// Data layer
class DataLayer : public BaseLayer {
public:
    DataLayer(LayerOption opt_);
    void Forward(Tensor *input_tensor_);
    inline const char* Type() const {return "Data"; }
};

// Convolution layer
class ConvolutionLayer : public BaseLayer {
public:
    ConvolutionLayer(LayerOption opt_);
    Tensor* connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace_);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "Convolution"; }
private:
    float *workspace;
};

// Pooling layer
class PoolingLayer : public BaseLayer {
public:
    PoolingLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "Pooling"; }
};

// AvgPooling layer
class AvgPoolingLayer : public BaseLayer {
public:
    AvgPoolingLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "AvgPooling"; }
};

// UpSample layer
class UpSampleLayer : public BaseLayer {
public:
    UpSampleLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "UpSample"; }
private:
    void upsample(float *in, int w, int h, int c, int batch, int stride, bool forward, float scale, float *out);
    void downsample(float *src, float *dst, int batch_size, int width, int height, int dimension, int stride, bool forward);
};

// Sigmoid layer
class SigmoidLayer : public BaseLayer {
public:
    SigmoidLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "Sigmoid"; }
};

// Tanh layer
class TanhLayer : public BaseLayer {
public:
    TanhLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "Tanh"; }
};

// Relu layer
class ReluLayer : public BaseLayer {
public:
    ReluLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "Relu"; }
};

// PRelu layer
class PReluLayer : public BaseLayer {
public:
    PReluLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "PRelu"; }
};

// LRelu layer
class LReluLayer : public BaseLayer {
public:
    LReluLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "LRelu"; }
};

// Mish layer
class MishLayer : public BaseLayer {
public:
    MishLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "Mish"; }
};

// Swish layer
class SwishLayer : public BaseLayer {
public:
    SwishLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "Swish"; }
};

// Elu layer
class EluLayer : public BaseLayer {
public:
    EluLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "Elu"; }
};

// Dropout layer
class DropoutLayer : public BaseLayer {
public:
    DropoutLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "Dropout"; }
};

// FullyConnected layer
class FullyConnectedLayer : public BaseLayer {
public:
    FullyConnectedLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "FullyConnected"; }
};

// BatchNormalization layer
class BatchNormalizationLayer : public BaseLayer {
public:
    BatchNormalizationLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "BatchNorm"; }
private:
    void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
    void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
    void variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
    void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);
};

// Concat layer
class ConcatLayer : public BaseLayer {
public:
    ConcatLayer(LayerOption opt_);
    Tensor* connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "Concat"; }
private:
    vtensorptr concat_tensor;
};

// Eltwise layer
class EltwiseLayer : public BaseLayer {
public:
    enum ELTWISE_OP {PROD = 0, SUM = 1, MAX = 2};
    EltwiseLayer(LayerOption opt_);
    Tensor* connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "Eltwise"; }
private:
    vtensorptr eltwise_tensor;
};

// ShortCut layer
class ShortCutLayer : public BaseLayer {
public:
    ShortCutLayer(LayerOption opt_);
    Tensor* connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "ShortCut"; }
private:
    void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);
    Tensor *shortcut_tensor;
};

// ScaleChannel layer
class ScaleChannelLayer : public BaseLayer {
public:
    ScaleChannelLayer(LayerOption opt_);
    Tensor *connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace);
    void Forward(bool train = false);
    void Backward(Tensor *none = nullptr);
    inline const char* Type() const {return "ScaleChannel"; }
private:
    Tensor *scalechannel_tensor;
};

// Softmax layer with cross entropy loss
class SoftmaxLayer : public BaseLayer {
public:
    SoftmaxLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *target);
    inline const char* Type() const {return "Softmax"; }
private:
};

// Euclidean loss layer
class EuclideanLossLayer : public BaseLayer {
public:
    EuclideanLossLayer(LayerOption opt_);
    void Forward(bool train = false);
    void Backward(Tensor *target);
    inline const char* Type() const {return "EuclideanLoss"; }
private:
};

// YOLOv3 layer
class YOLOv3Layer : public BaseLayer {
public:
    YOLOv3Layer(LayerOption opt_);
    Tensor* connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace);
    void Forward(bool train = false);
    void Backward(Tensor *target);
    inline const char* Type() const {return "YOLOv3"; }
private:
    int entry_index(int batch, int location, int entry);
    vector<Detection> yolo_get_detection_without_correction();
    Box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride);
    float delta_yolo_box(Box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride);
    void delta_yolo_class(float *output, float *delta, int index, int cls, int classes, int stride, float *avg_cat);
    int int_index(float *a, int val, int n);
    float mag_array(float *a, int n);
    
    Tensor detection;
};

typedef struct train_yolo_args {
    BaseLayer::Info info;
    float *output;
    float *delta;
    int b;

    float tot_iou;
    float tot_giou_loss;
    float tot_iou_loss;
    int count;
    int class_count;
} train_yolo_args;

// YOLOv4 layer
class YOLOv4Layer : public BaseLayer {
public:
    YOLOv4Layer(LayerOption opt_);
    Tensor* connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace);
    void Forward(bool train = false);
    void Backward(Tensor *target);
    inline const char* Type() const {return "YOLOv4"; }
private:
    int entry_index(int batch, int location, int entry);
    vector<Detection> yolo_get_detection_without_correction();
    Box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride, bool new_coordinate);
    Ious delta_yolo_box(Box &truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride, float iou_normalizer, IOU_KIND iou_loss, int accumulate, float max_delta, int *rewritten_bbox, int new_coords);
    void delta_yolo_class(float *output, float *delta, int index, int class_id, int classes, int stride, float *avg_cat, int focal_loss, float label_smooth_eps, float *classes_multipliers, float cls_normalizer);
    int compare_yolo_class(float *output, int classes, int class_index, int stride, float objectness, int class_id, float conf_thresh);
    void averages_yolo_deltas(int class_index, int box_index, int stride, int classes, float *delta);
    int int_index(float *a, int val, int n);
    float mag_array(float *a, int n);
    
    Tensor detection;
};

void scale_bias(float *output, float *scales, int batch, int n, int size);
void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

#endif /* Layer_hpp */
