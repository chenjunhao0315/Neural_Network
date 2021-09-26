//
//  Box.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/9/25.
//

#ifndef Box_hpp
#define Box_hpp

#include <stdio.h>
#include <float.h>
#include <vector>
#include <math.h>

using namespace std;

typedef vector<float> vfloat;

enum IOU_KIND {
    IOU,
    GIOU,
    DIOU,
    CIOU,
    MSE
};

typedef struct Box {
    float x, y, w, h;
} Box;

typedef struct Boxabs {
    float left, right, top, bot;
} Boxabs;

typedef struct Dxrep {
    float dt, db, dl, dr;
} Dxrep;

typedef struct Ious {
    float iou, giou, diou, ciou;
    Dxrep dx_iou;
    Dxrep dx_giou;
} Ious;

struct Detection {
    Box bbox;
    int classes;
    vector<float> prob;
    float objectness;
    int sort_class;
};

float overlap(float x1, float w1, float x2, float w2);
float box_intersection(Box &a, Box &b);
float box_union(Box &a, Box &b);
float box_iou_kind(Box &a, Box &b, IOU_KIND iou_kind);
float box_iou(Box &a, Box &b);
float box_giou(Box &a, Box &b);
float box_diou(Box &a, Box &b);
float box_ciou(Box &a, Box &b);
Boxabs box_c(Box &a, Box &b);
Boxabs to_tblr(Box &a);
Dxrep dx_box_iou(Box &pred, Box &truth, IOU_KIND iou_loss);

Box float_to_box(float *f, int stride);
Box vfloat_to_box(vfloat &src, int index);

static inline float fix_nan_inf(float val) {
    if (isnan(val) || isinf(val)) val = 0;
    return val;
}

static inline float clip_value(float val, const float max_val) {
    if (val > max_val) {
        val = max_val;
    }
    else if (val < -max_val) {
        val = -max_val;
    }
    return val;
}

#endif /* Box_hpp */
