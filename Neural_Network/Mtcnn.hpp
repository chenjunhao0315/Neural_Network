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
#include "Neural_Network.hpp"
#include "Image_Process.hpp"


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
    Bbox(int x1_, int y1_, int x2_, int y2_, float s, float dx1_, float dy1_, float dx2_, float dy2_): x1(x1_), y1(y1_), x2(x2_), y2(y2_), score(s), dx1(dx1_), dy1(dy1_), dx2(dx2_), dy2(dy2_) {};
};

class Mtcnn {
    Mtcnn(const char *model_pnet, const char *model_rnet, const char *model_onet);
    vector<Bbox> detect(IMG &img);
    Neural_Network pnet;
    Neural_Network rnet;
    Neural_Network onet;
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

vector<Bbox> generate_bbox(Feature_map map, float scale, float threshold);
vector<Bbox> nms(vector<Bbox> &Bbox_list, float threshold);
vector<Bbox> convert_to_square(vector<Bbox> &Bbox_list);
vector<Bbox> crop_coordinate(vector<Bbox> &Bbox_list, int width, int height);
float iou(Bbox box1, Bbox box2);

void mtcnn_data_loader(const char *data_path, const char *label_path, vtensor &data_set, vector<vfloat> &label_set, int width, int size);
void mtcnn_evaluate(Neural_Network *nn, vtensor &data_set, vector<vfloat> &label_set);

#endif /* Mtcnn_hpp */
