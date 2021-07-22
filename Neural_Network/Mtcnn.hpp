//
//  Mtcnn.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/7/19.
//

#ifndef Mtcnn_hpp
#define Mtcnn_hpp

#include <stdio.h>
#include <vector>
#include <chrono>
#include "Neural_Network.hpp"
#include "Image_Process.hpp"

using namespace std::chrono;

typedef vector<vector<vfloat>> Feature_map;
typedef vector<float> Box;

struct Bbox {
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    float dx1;
    float dy1;
    float dx2;
    float dy2;
    float lefteye_x;
    float lefteye_y;
    float righteye_x;
    float righteye_y;
    float nose_x;
    float nose_y;
    float leftmouth_x;
    float leftmouth_y;
    float rightmouth_x;
    float rightmouth_y;
    Bbox(int x1_, int y1_, int x2_, int y2_, float s, float dx1_, float dy1_, float dx2_, float dy2_): x1(x1_), y1(y1_), x2(x2_), y2(y2_), score(s), dx1(dx1_), dy1(dy1_), dx2(dx2_), dy2(dy2_) {};
    Bbox(int x1_, int y1_, int x2_, int y2_, float s, float dx1_, float dy1_, float dx2_, float dy2_, float lefteye_x_, float lefteye_y_, float righteye_x_, float righteye_y_, float nose_x_, float nose_y_, float leftmouth_x_, float leftmouth_y_, float rightmouth_x_, float rightmouth_y_): x1(x1_), y1(y1_), x2(x2_), y2(y2_), score(s), dx1(dx1_), dy1(dy1_), dx2(dx2_), dy2(dy2_), lefteye_x(lefteye_x_), lefteye_y(lefteye_y_), righteye_x(righteye_x_), righteye_y(righteye_y_), nose_x(nose_x_), nose_y(nose_y_), leftmouth_x(leftmouth_x_), leftmouth_y(leftmouth_y_), rightmouth_x(rightmouth_x_), rightmouth_y(rightmouth_y_) {};
};

class PNet {
public:
    PNet();
    PNet(const char *model_name);
    ~PNet() {
        pnet.save("pnet_ensure.bin");
    }
    vector<Bbox> detect(IMG &img);
    Feature_map predict(IMG &img);
    //private:
    Neural_Network pnet;
    int min_face_size;
    float scale_factor;
    float threshold[3];
};

class RNet {
public:
    ~RNet() {
        rnet.save("rnet_ensure.bin");
    }
    RNet();
    RNet(const char *model_name);
    vector<Bbox> detect(IMG &img, vector<Bbox> &pnet_bbox);
    Neural_Network rnet;
    float threshold[3];
};

class ONet {
public:
    ~ONet() {
        onet.save("onet_ensure.bin");
    }
    ONet();
    ONet(const char *model_name);
    vector<Bbox> detect(IMG &img, vector<Bbox> &rnet_bbox);
    Neural_Network onet;
    float threshold[3];
};

class Mtcnn {
public:
    ~Mtcnn();
    Mtcnn();
    Mtcnn(const char *model_pnet, const char *model_rnet, const char *model_onet);
    vector<Bbox> detect(IMG &img);
    PNet *pnet;
    RNet *rnet;
    Neural_Network onet;
};

vector<Bbox> generate_bbox(Feature_map map, float scale, float threshold);
vector<Bbox> nms(vector<Bbox> &Bbox_list, float threshold, int mode = 0);
vector<Bbox> convert_to_square(vector<Bbox> &Bbox_list);
vector<Bbox> crop_coordinate(vector<Bbox> &Bbox_list, int width, int height);
float iou(Bbox box1, Bbox box2, int mode);

void mtcnn_data_loader(const char *data_path, const char *label_path, vtensor &data_set, vector<vfloat> &label_set, int width, int size);
void mtcnn_evaluate(Neural_Network *nn, vtensor &data_set, vector<vfloat> &label_set);

#endif /* Mtcnn_hpp */
