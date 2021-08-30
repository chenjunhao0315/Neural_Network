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
    YOLOv3(const char *model_name);
    YOLOv3(int num_class_ = 80);
    vector<Detection> detect(IMG &input);
//private:
    IMG yolo_pre_process_img(IMG& img, int net_w, int net_h);
    vector<Detection> yolo_correct_box(Tensor *box_list, int img_w, int img_h, int net_w, int net_h, bool relative);
    void yolo_nms(vector<Detection> &list, int classes ,float threshold);
    
    Neural_Network network;
    int classes;
    float threshold;
    int net_width, net_height;
};

void yolo_mark(vector<Detection> &dets, IMG &img, int classes, float threshold);

struct yolo_label {
    yolo_label() {}
    yolo_label(vector<Detection> det_) : det(det_) {}
    string filename;
    vector<Detection> det;
};

class YOLOv3_DataLoader {
public:
    ~YOLOv3_DataLoader();
    YOLOv3_DataLoader(const char *filename);
    yolo_label get_label(int index) {return dataset[index];}
    void mark_truth(int index);
    int size() {return (int)dataset.size() - 1;}
private:
    vector<yolo_label> dataset;
};

#endif /* YOLOv3_hpp */
