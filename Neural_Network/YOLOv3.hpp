//
//  YOLOv3.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/8/24.
//

#ifndef YOLOv3_hpp
#define YOLOv3_hpp

#include <stdio.h>
#include "Neural_Network.hpp"
#include "Image_Process.hpp"

class YOLOv3 {
public:
    YOLOv3(const char *model_name, int num_class_ = 80, int batch_size = 1);
    YOLOv3(int num_class_ = 80, int batch_size = 1);
    vector<Detection> detect(IMG &input);
//private:
    IMG yolo_pre_process_img(IMG& img, int net_w, int net_h);
    vector<Detection> yolo_correct_box(Tensor *box_list, int img_w, int img_h, int net_w, int net_h, bool relative);
    void yolo_nms(vector<Detection> &list, int classes ,float threshold);
    vector<string> get_yolo_label(const char *labelstr, int classes);
    
    Neural_Network network;
    int classes;
    float threshold;
    int net_width, net_height;
    vector<string> label;
};

void yolo_mark(vector<Detection> &dets, IMG &img, int classes, float threshold, vector<string> label = vector<string>());

struct Box_label {
    int id;
    float x, y, w, h;
    float left, right, top, bottom;
};

struct yolo_label {
    yolo_label() {}
    yolo_label(vector<Box_label> boxes_) : boxes(boxes_) {}
    string filename;
    vector<Box_label> boxes;
};

struct yolo_train_args {
    yolo_train_args(Tensor &data_, Tensor &label_) : data(data_), label(label_) {}
    Tensor data;
    Tensor label;
};

class YOLOv3_DataLoader {
public:
    ~YOLOv3_DataLoader();
    YOLOv3_DataLoader(const char *filename);
    void clean_data(const char *in, const char *out);
    yolo_train_args get_train_arg(int index);
    void mark_truth(int index);
    int size() {return (int)dataset.size() - 1;}
//private:
    IMG get_img(int index);
    vector<Box_label> get_label(int index);
    vfloat get_box(int index, float dx, float dy, float sx, float sy, bool flip, int net_w, int net_h);
    void correct_box(vector<Box_label> &boxes, float dx, float dy, float sx, float sy, bool flip);
    vector<yolo_label> dataset;
};

class YOLOv3_Trainer {
public:
    YOLOv3_Trainer(Neural_Network *network_, Trainer *trainer_, YOLOv3_DataLoader *loader_) : network(network_), trainer(trainer_), loader(loader_) {}
    void train(int epoch);
private:
    Neural_Network *network;
    Trainer *trainer;
    YOLOv3_DataLoader *loader;
};

#endif /* YOLOv3_hpp */
