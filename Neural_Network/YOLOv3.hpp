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

struct Box {
    int x, y, w, h;
};

struct Detection {
    Box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
};

class YOLOv3 {
public:
    YOLOv3(int num_class_ = 80);
private:
    Neural_Network network;
    int num_class;
};

#endif /* YOLOv3_hpp */
