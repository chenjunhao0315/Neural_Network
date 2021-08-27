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
    IMG pre_process_img(IMG& img, int net_w, int net_h);
    vector<Detection> correct_box(Tensor *box_list, int img_w, int img_h, int net_w, int net_h, bool relative);
    void yolo_nms(vector<Detection> &list, int classes ,float threshold);
    
    Neural_Network network;
    int classes;
    float threshold;
    int net_width, net_height;
};

void yolo_mark(vector<Detection> &dets, IMG &img, int classes, float threshold);

#endif /* YOLOv3_hpp */
