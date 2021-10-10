//
//  Mtcnn.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/7/19.
//

#include "Mtcnn.hpp"

PNet::PNet(const char *model_name) {
    scale_factor = 0.709;
    threshold[0] = 0.97;
    pnet = Neural_Network("mtcnn");
//    pnet.load(model_name);
    pnet.load_ottermodel(model_name);
//    pnet.shape();
}

PNet::PNet(int batch_size) {
    scale_factor = 0.709;
    threshold[0] = 0.97;
    pnet = Neural_Network("mtcnn");
    pnet.addLayer(LayerOption{{"type", "Input"}, {"input_width", "12"}, {"input_height", "12"}, {"input_dimension", "3"}, {"name", "input"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "10"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_1"}});
    pnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_1"}});
    pnet.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}, {"name", "pool_1"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_2"}});
    pnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_2"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_3"}});
    pnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_3"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "2"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_4"}});
    pnet.addLayer(LayerOption{{"type", "Softmax"}, {"name", "cls_prob"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "4"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_5"}, {"input_name", "prelu_3"}});
    pnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "bbox_pred"}, {"input_name", "conv_5"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "10"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_6"}, {"input_name", "prelu_3"}});
    pnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "land_pred"}, {"input_name", "conv_6"}});
    pnet.addOutput("cls_prob");
    pnet.addOutput("bbox_pred");
    pnet.addOutput("land_pred");
    pnet.compile(batch_size);
    pnet.shape();
}

vector<Bbox> PNet::detect(IMG &img, int min_face_size) {
    int net_size = 12;
//    float current_scale = float(net_size) / min_face_size;
//    float current_scale = (float)1000 / max(img.width, img.height);
    float current_scale = 1.0;
    if (min_face_size < 12) {
        if (min(img.width, img.height) > 650) {
            current_scale = 650.0 / min(img.height, img.width);
        } else if (max(img.width, img.height) < 650) {
            current_scale = 650.0 / max(img.height, img.width);
        }
    } else {
        current_scale = float(net_size) / min_face_size;
    }
    
    IMG img_resize = img.resize(Size(), current_scale, current_scale);
    int current_width = img_resize.width;
    int current_height = img_resize.height;
    vector<Bbox> bbox_list;
    
    while(min(current_width, current_height) > net_size) {
        Feature_map map = this->predict(img_resize);
        vector<Bbox> list = generate_bbox(map, current_scale, threshold[0]);
        list = nms(list, 0.5);
        bbox_list.insert(bbox_list.end(), list.begin(), list.end());
        
        current_scale *= scale_factor;
        img_resize = img.resize(Size(), current_scale, current_scale);
        current_width = img_resize.width;
        current_height = img_resize.height;
    }
    
    bbox_list = nms(bbox_list, 0.7);
    
    for (int i = 0; i < bbox_list.size(); ++i) {
        int bbox_w = bbox_list[i].x2 - bbox_list[i].x1 + 1;
        int bbox_h = bbox_list[i].y2 - bbox_list[i].y1 + 1;
        bbox_list[i].x1 += bbox_list[i].dx1 * bbox_w;
        bbox_list[i].x2 += bbox_list[i].dx2 * bbox_w;
        bbox_list[i].y1 += bbox_list[i].dy1 * bbox_h;
        bbox_list[i].y2 += bbox_list[i].dy2 * bbox_h;
    }
    return bbox_list;
}

Feature_map PNet::predict(IMG &img) {
    int width = img.width;
    int height = img.height;
    int stride = 2;
    int padding = 0;
    int map_width = (width + padding * 2 - 12) / stride + 1;
    int map_height = (height + padding * 2 - 12) / stride + 1;
    Feature_map map(map_height, vector<vfloat>(map_width, vfloat()));
    int x, y;
    int h, w, f_h, f_w, f_d;
    int coordinate_h, coordinate_w;
    Tensor input(12, 12, 3, 0);
    
    unsigned char *rgb = img.toPixelArray();
    int img_size = img.width * img.height * 3;
    float *img_data = new float [img_size];
    for (int i = 0; i < img_size; ++i) {
        img_data[i] = ((float)rgb[i] - 127.5) * 0.0078125;
    }
    float *data = input.weight;
    
    y = -padding;
    for (h = 0; h < map_height; ++h, y += stride) {
        x = -padding;
        for (w = 0; w < map_width; ++w, x += stride) {
            for (f_d = 0; f_d < 3; ++f_d) {
                for (f_h = 0; f_h < 12; ++f_h) {
                    coordinate_h = y + f_h;
                    for (f_w = 0; f_w < 12; ++f_w) {
                        coordinate_w = x + f_w;
                        if (coordinate_w >= 0 && coordinate_w < width && coordinate_h >= 0 && coordinate_h < height)
                            data[((12 * f_h) + f_w) + 144 * f_d] = img_data[((width * coordinate_h) + coordinate_w) * 3 + f_d];
                        else
                            data[((12 * f_h) + f_w) + 144 * f_d] = 0;
                    }
                }
            }
            vtensorptr output = pnet.Forward(&input);
            vfloat feature; feature.reserve(6);
            for (int i = 0; i < 2; ++i) {
                vfloat extract = output[i]->toVector();
                feature.insert(feature.end(), extract.begin(), extract.end());
            }
            map[h][w] = feature;
        }
    }
    delete [] img_data;
    return map;
}

bool PNet::ready() {
    return (pnet.status() == Neural_Network::nn_status::OK) ? true : false;
}

vector<Bbox> generate_bbox(Feature_map map, float scale, float threshold) {
    int stride = 2;
    int cellsize = 12;
    int height = (int)map.size();
    int width = (int)map[0].size();
    vector<Bbox> bbox_list;
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (map[i][j][1] > threshold) {
                int x1 = (int)round(stride * j / scale);
                int y1 = (int)round(stride * i / scale);
                int x2 = (int)round((stride * j + cellsize) / scale);
                int y2 = (int)round((stride * i + cellsize) / scale);
                float dx1 = map[i][j][2];
                float dy1 = map[i][j][3];
                float dx2 = map[i][j][4];
                float dy2 = map[i][j][5];
                Bbox bbox(x1, y1, x2, y2, map[i][j][1], dx1, dy1, dx2, dy2);
                bbox_list.push_back(bbox);
            }
        }
    }
    return bbox_list;
}

vector<Bbox> nms(vector<Bbox> &Bbox_list, float threshold, int mode) {
    auto cmpScore = [](Bbox &box1, Bbox &box2) {
        return box1.score < box2.score;
    };
    sort(Bbox_list.begin(), Bbox_list.end(), cmpScore);
    
    vector<Bbox> pickedBbox;
    while (Bbox_list.size() > 0) {
        pickedBbox.emplace_back(Bbox_list.back());
        Bbox_list.pop_back();
        for (size_t i = 0; i < Bbox_list.size(); i++) {
            if (iou(pickedBbox.back(), Bbox_list[i], mode) >= threshold) {
                Bbox_list.erase(Bbox_list.begin() + i);
            }
        }
    }
    return pickedBbox;
}

float iou(Bbox box1, Bbox box2, int mode) {
    int area1 = (box1.x2 - box1.x1 + 1) * (box1.y2 - box1.y1 + 1);
    int area2 = (box2.x2 - box2.x1 + 1) * (box2.y2 - box2.y1 + 1);
    int x11 = max(box1.x1, box2.x1);
    int y11 = max(box1.y1, box2.y1);
    int x22 = min(box1.x2, box2.x2);
    int y22 = min(box1.y2, box2.y2);
    int width = ((x22 - x11 + 1) > 0) ? (x22 - x11 + 1) : 0;
    int height = ((y22 - y11 + 1) > 0) ? (y22 - y11 + 1) : 0;
    float intersection = width * height;
    if (mode) {
        return intersection / min(area1, area2);
    } else {
        return intersection / (area1 + area2 - intersection);
    }
    return 0;
}

vector<Bbox> convert_to_square(vector<Bbox> &Bbox_list) {
    vector<Bbox> square_bbox = Bbox_list;
    for (int i = 0; i < square_bbox.size(); ++i) {
        int h = Bbox_list[i].y2 - Bbox_list[i].y1 + 1;
        int w = Bbox_list[i].x2 - Bbox_list[i].x1 + 1;
        int max_side = (h > w) ? h : w;
        square_bbox[i].x1 = round(Bbox_list[i].x1 + w * 0.5 - max_side * 0.5);
        square_bbox[i].y1 = round(Bbox_list[i].y1 + h * 0.5 - max_side * 0.5);
        square_bbox[i].x2 = round(square_bbox[i].x1 + max_side - 1);
        square_bbox[i].y2 = round(square_bbox[i].y1 + max_side - 1);
    }
    return square_bbox;
}

vector<Bbox> calibrate_box(vector<Bbox> &Bbox_list) {
    vector<Bbox> bbox_c = Bbox_list;
    size_t Bbox_c_size = bbox_c.size();
    for (int i = 0; i < Bbox_c_size; ++i) {
        int width = bbox_c[i].x2 - bbox_c[i].x1 + 1;
        int height = bbox_c[i].y2 - bbox_c[i].y1 + 1;
        bbox_c[i].x1 += bbox_c[i].dx1 * width;
        bbox_c[i].y1 += bbox_c[i].dy1 * height;
        bbox_c[i].x2 += bbox_c[i].dx2 * width;
        bbox_c[i].y2 += bbox_c[i].dy2 * height;
    }
    return bbox_c;
}

RNet::RNet(const char *model_name) {
    rnet = Neural_Network("mtcnn");
//    rnet.load(model_name);
    rnet.load_ottermodel(model_name);
//    rnet.shape();
}

RNet::RNet() {
    threshold[0] = 0.9;
    rnet = Neural_Network("mtcnn");
    rnet.addLayer(LayerOption{{"type", "Input"}, {"input_width", "24"}, {"input_height", "24"}, {"input_dimension", "3"}, {"name", "input"}});
    rnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "28"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_1"}});
    rnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_1"}});
    rnet.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "3"}, {"stride", "2"}, {"name", "pool_1"}, {"padding", "1"}});
    rnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "48"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_2"}});
    rnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_2"}});
    rnet.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "3"}, {"stride", "2"}, {"name", "pool_2"}});
    rnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "2"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_3"}});
    rnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_3"}});
    rnet.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "128"}, {"name", "fc_1"}});
    rnet.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "2"}, {"name", "fc_cls"}});
    rnet.addLayer(LayerOption{{"type", "Softmax"}, {"name", "cls_prob"}});
    rnet.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "4"}, {"name", "fc_bbox"}, {"input_name", "fc_1"}});
    rnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "bbox_pred"}, {"input_name", "fc_bbox"}});
    rnet.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "10"}, {"name", "fc_land"}, {"input_name", "fc_1"}});
    rnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "land_pred"}, {"input_name", "fc_land"}});
    rnet.addOutput("cls_prob");
    rnet.addOutput("bbox_pred");
    rnet.addOutput("land_pred");
    rnet.compile();
    rnet.shape();
}

vector<Bbox> RNet::detect(IMG &img, vector<Bbox> &pnet_bbox) {
    vector<Bbox> square_bbox = convert_to_square(pnet_bbox);
    vector<Bbox> rnet_bbox;
    size_t square_bbox_size = square_bbox.size();
    Tensor crop_img(24, 24, 3, 0);
    for (int i = 0; i < square_bbox_size; ++i) {
        IMG crop = img.crop(Rect(square_bbox[i].x1, square_bbox[i].y1, square_bbox[i].x2, square_bbox[i].y2));
        crop = crop.resize(Size(24, 24));
        unsigned char *pixel = crop.toPixelArray();
        float *pixel_c = crop_img.weight;
        int count = 0;
        for (int d = 0; d < 3; ++d) {
            for (int h = 0; h < 24; ++h) {
                for (int w = 0; w < 24; ++w) {
                    pixel_c[count++] = ((float)pixel[((24 * h) + w) * 3 + d] - 127.5) * 0.0078125;
                }
            }
        }
        
        vtensorptr output = rnet.Forward(&crop_img);
        vfloat rnet_detect; rnet_detect.reserve(16);
        for (int i = 0; i < 3; ++i) {
            vfloat extract = output[i]->toVector();
            rnet_detect.insert(rnet_detect.end(), extract.begin(), extract.end());
        }
        if (rnet_detect[1] > threshold[0]) {
            Bbox rnet_detect_box(square_bbox[i].x1, square_bbox[i].y1, square_bbox[i].x2, square_bbox[i].y2, rnet_detect[1], rnet_detect[2], rnet_detect[3], rnet_detect[4], rnet_detect[5]);
            rnet_bbox.push_back(rnet_detect_box);
        }
    }
    rnet_bbox = nms(rnet_bbox, 0.6);
    rnet_bbox = calibrate_box(rnet_bbox);
    return rnet_bbox;
}

bool RNet::ready() {
    return (rnet.status() == Neural_Network::nn_status::OK) ? true : false;
}

ONet::ONet(const char *model_name) {
    onet = Neural_Network("mtcnn");
//    onet.load(model_name);
    onet.load_ottermodel(model_name);
//    onet.shape();
}

ONet::ONet() {
    threshold[0] = 0.9;
    onet = Neural_Network("mtcnn");
    onet.addLayer(LayerOption{{"type", "Input"}, {"input_width", "48"}, {"input_height", "48"}, {"input_dimension", "3"}, {"name", "input"}});
    onet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_1"}});
    onet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_1"}});
    onet.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "3"}, {"stride", "2"}, {"name", "pool_1"}, {"padding", "1"}});
    onet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_2"}});
    onet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_2"}});
    onet.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "3"}, {"stride", "2"}, {"name", "pool_2"}});
    onet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_3"}});
    onet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_3"}});
    onet.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}, {"name", "pool_3"}});
    onet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "2"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_4"}});
    onet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_4"}});
    onet.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "256"}, {"name", "fc_1"}});
    onet.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "2"}, {"name", "fc_cls"}});
    onet.addLayer(LayerOption{{"type", "Softmax"}, {"name", "cls_prob"}});
    onet.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "4"}, {"name", "fc_bbox"}, {"input_name", "fc_1"}});
    onet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "bbox_pred"}, {"input_name", "fc_bbox"}});
    onet.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "10"}, {"name", "fc_land"}, {"input_name", "fc_1"}});
    onet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "land_pred"}, {"input_name", "fc_land"}});
    onet.addOutput("cls_prob");
    onet.addOutput("bbox_pred");
    onet.addOutput("land_pred");
    onet.compile();
    onet.shape();
}

vector<Bbox> ONet::detect(IMG &img, vector<Bbox> &rnet_bbox) {
    vector<Bbox> square_bbox = convert_to_square(rnet_bbox);
    vector<Bbox> onet_bbox;
    size_t square_bbox_size = square_bbox.size();
    for (int i = 0; i < square_bbox_size; ++i) {
        IMG crop = img.crop(Rect(square_bbox[i].x1, square_bbox[i].y1, square_bbox[i].x2, square_bbox[i].y2));
        crop = crop.resize(Size(48, 48));
        unsigned char *pixel = crop.toPixelArray();
        Tensor crop_img(48, 48, 3, 0);
        float *pixel_c = crop_img.weight;
        int count = 0;
        for (int d = 0; d < 3; ++d) {
            for (int h = 0; h < 48; ++h) {
                for (int w = 0; w < 48; ++w) {
                    pixel_c[count++] = ((float)pixel[((48 * h) + w) * 3 + d] - 127.5) * 0.0078125;
                }
            }
        }
        
        vtensorptr output = onet.Forward(&crop_img);
        vfloat onet_detect; onet_detect.reserve(16);
        for (int i = 0; i < 3; ++i) {
            vfloat extract = output[i]->toVector();
            onet_detect.insert(onet_detect.end(), extract.begin(), extract.end());
        }
        if (onet_detect[1] > threshold[0]) {
            Bbox rnet_detect_box(square_bbox[i].x1, square_bbox[i].y1, square_bbox[i].x2, square_bbox[i].y2, onet_detect[1], onet_detect[2], onet_detect[3], onet_detect[4], onet_detect[5], onet_detect[6], onet_detect[7], onet_detect[8], onet_detect[9], onet_detect[10], onet_detect[11], onet_detect[12], onet_detect[13], onet_detect[14], onet_detect[15]);
            onet_bbox.push_back(rnet_detect_box);
        }
    }
    
    for (int i = 0; i < onet_bbox.size(); ++i) {
        int bbox_w = onet_bbox[i].x2 - onet_bbox[i].x1 + 1;
        int bbox_h = onet_bbox[i].y2 - onet_bbox[i].y1 + 1;
        onet_bbox[i].lefteye_x = round(onet_bbox[i].lefteye_x * bbox_w + onet_bbox[i].x1 - 1);
        onet_bbox[i].lefteye_y = round(onet_bbox[i].lefteye_y * bbox_h + onet_bbox[i].y1 - 1);
        onet_bbox[i].righteye_x = round(onet_bbox[i].righteye_x * bbox_w + onet_bbox[i].x1 - 1);
        onet_bbox[i].righteye_y = round(onet_bbox[i].righteye_y * bbox_h + onet_bbox[i].y1 - 1);
        onet_bbox[i].nose_x = round(onet_bbox[i].nose_x * bbox_w + onet_bbox[i].x1 - 1);
        onet_bbox[i].nose_y = round(onet_bbox[i].nose_y * bbox_h + onet_bbox[i].y1 - 1);
        onet_bbox[i].leftmouth_x = round(onet_bbox[i].leftmouth_x * bbox_w + onet_bbox[i].x1 - 1);
        onet_bbox[i].leftmouth_y = round(onet_bbox[i].leftmouth_y * bbox_h + onet_bbox[i].y1 - 1);
        onet_bbox[i].rightmouth_x = round(onet_bbox[i].rightmouth_x * bbox_w + onet_bbox[i].x1 - 1);
        onet_bbox[i].rightmouth_y = round(onet_bbox[i].rightmouth_y * bbox_h + onet_bbox[i].y1 - 1);
    }
    
    onet_bbox = calibrate_box(onet_bbox);
    onet_bbox = nms(onet_bbox, 0.7, 1);
    
    return onet_bbox;
}

bool ONet::ready() {
    return (onet.status() == Neural_Network::nn_status::OK) ? true : false;
}

Mtcnn::~Mtcnn() {
    delete pnet;
    delete rnet;
    delete onet;
}

Mtcnn::Mtcnn(const char *model_pnet, const char *model_rnet, const char *model_onet) {
    pnet = new PNet(model_pnet);
    rnet = new RNet(model_rnet);
    onet = new ONet(model_onet);
    if ((pnet->ready() && rnet->ready() && onet->ready()))
        printf("[Mtcnn] Network Ready!\n");
    else
        return;
    pnet->threshold[0] = 0.9;
    rnet->threshold[0] = 0.7;
    onet->threshold[0] = 0.8;
    min_face_size = 0;
}

vector<Bbox> Mtcnn::detect(IMG &img) {
    if (!(pnet->ready() && rnet->ready() && onet->ready())) {
        printf("[Mtcnn] Network error!\n");
        return vector<Bbox>();
    }
    if (img.getMat().depth() != 3) {
        printf("[Mtcnn] Image format unsupport!\n");
        return vector<Bbox>();
    }
    
    Clock c;
    vector<Bbox> bbox = pnet->detect(img, min_face_size);
    
    printf("PNet Get %d proposal box! time: %lldms\n", (int)bbox.size(), c.getElapsed());

    IMG pnet_detect(img);
    for (int i = 0; i < bbox.size(); ++i) {
        pnet_detect.drawRectangle(Rect{(bbox[i].x1), (bbox[i].y1), (bbox[i].x2), (bbox[i].y2)}, RED);
    }
    pnet_detect.save("pnet_predict.jpg", 80);

    c.start();
    vector<Bbox> rnet_bbox = rnet->detect(img, bbox);
    printf("RNet Get %d proposal box! time: %lldms\n", (int)rnet_bbox.size(), c.getElapsed());

    IMG rnet_detect(img);
    for (int i = 0; i < rnet_bbox.size(); ++i) {
        rnet_detect.drawRectangle(Rect{(rnet_bbox[i].x1), (rnet_bbox[i].y1), (rnet_bbox[i].x2), (rnet_bbox[i].y2)}, Color(255, 0, 0));
    }
    rnet_detect.save("rnet_predict.jpg", 80);

    c.start();
    vector<Bbox> onet_bbox = onet->detect(img, rnet_bbox);
    printf("ONet Get %d proposal box! time: %lldms\n", (int)onet_bbox.size(), c.getElapsed());

    IMG onet_detect(img);
    for (int i = 0; i < onet_bbox.size(); ++i) {
        int radius = min(onet_bbox[i].x2 - onet_bbox[i].x1 + 1, onet_bbox[i].y2 - onet_bbox[i].y1 + 1) / 30 + 1;
        onet_detect.drawRectangle(Rect{(onet_bbox[i].x1), (onet_bbox[i].y1), (onet_bbox[i].x2), (onet_bbox[i].y2)}, RED);
        onet_detect.drawCircle(Point(onet_bbox[i].lefteye_x, onet_bbox[i].lefteye_y), RED, radius, radius);
        onet_detect.drawCircle(Point(onet_bbox[i].righteye_x, onet_bbox[i].righteye_y), RED, radius, radius);
        onet_detect.drawCircle(Point(onet_bbox[i].nose_x, onet_bbox[i].nose_y), RED, radius, radius);
        onet_detect.drawCircle(Point(onet_bbox[i].leftmouth_x, onet_bbox[i].leftmouth_y), RED, radius, radius);
        onet_detect.drawCircle(Point(onet_bbox[i].rightmouth_x, onet_bbox[i].rightmouth_y), RED, radius, radius);
    }
    onet_detect.save("onet_predict.jpg", 80);
    return onet_bbox;
}

void Mtcnn::mark(IMG &img, vector<Bbox> &bbox_list, bool confidence) {
    for (int i = 0; i < bbox_list.size(); ++i) {
        int radius = min(bbox_list[i].x2 - bbox_list[i].x1 + 1, bbox_list[i].y2 - bbox_list[i].y1 + 1) / 30 + 1;
        img.drawRectangle(Rect{(bbox_list[i].x1), (bbox_list[i].y1), (bbox_list[i].x2), (bbox_list[i].y2)}, RED);
        img.drawCircle(Point(bbox_list[i].lefteye_x, bbox_list[i].lefteye_y), RED, radius, radius);
        img.drawCircle(Point(bbox_list[i].righteye_x, bbox_list[i].righteye_y), RED, radius, radius);
        img.drawCircle(Point(bbox_list[i].nose_x, bbox_list[i].nose_y), RED, radius, radius);
        img.drawCircle(Point(bbox_list[i].leftmouth_x, bbox_list[i].leftmouth_y), RED, radius, radius);
        img.drawCircle(Point(bbox_list[i].rightmouth_x, bbox_list[i].rightmouth_y), RED, radius, radius);
        if (confidence) {
            string score = to_string(bbox_list[i].score).substr(0, 4);
            int h = img.height * 0.02;
            if (h > (bbox_list[i].y2 - bbox_list[i].y1 + 1) * 0.3)
                h = (bbox_list[i].y2 - bbox_list[i].y1 + 1) * 0.3;
            if (h < 22)
                h = 22;
            img.putText(score.c_str(), Point(bbox_list[i].x1, bbox_list[i].y1 - h), GREEN, h);
        }
    }
}

vector<vector<Bbox>> Mtcnn::detect(const char *filelist) {
    fstream list;
    list.open(filelist, ios::in);
    while(!list.eof()) {
        string filename;
        list >> filename;
        cout << "Processing: " << filename << endl;
        IMG img(filename.c_str());
        vector<Bbox> bbox_list = detect(img);
        mark(img, bbox_list);
        size_t pos = filename.find(".");
        filename = filename.substr(0, pos) + "_detected.jpg";
        img.save(filename.c_str(), 80);
    }
    return vector<vector<Bbox>>();
}

void Mtcnn::layout(vector<Bbox> &bbox_list, const char *filename) {
    FILE *f = fopen(filename, "w");
    int size = (int)bbox_list.size();
    fprintf(f, "[\n");
    for (int i = 0; i < size; ++i) {
        Bbox &box = bbox_list[i];
        fprintf(f, "\t{\n");
        fprintf(f, "\t\t'box': [%d, %d, %d, %d],\n", box.x1, box.y1, box.x2, box.y2);
        fprintf(f, "\t\t'keypoints':\n\t\t{\n");
        fprintf(f, "\t\t\t'eye_right': (%d, %d),\n", (int)box.righteye_x, (int)box.righteye_y);
        fprintf(f, "\t\t\t'eye_left': (%d, %d),\n", (int)box.lefteye_x, (int)box.lefteye_y);
        fprintf(f, "\t\t\t'nose': (%d, %d),\n", (int)box.nose_x, (int)box.nose_y);
        fprintf(f, "\t\t\t'mouth_right': (%d, %d),\n", (int)box.rightmouth_x, (int)box.rightmouth_y);
        fprintf(f, "\t\t\t'mouth_left': (%d, %d)\n\t\t},\n", (int)box.leftmouth_x, (int)box.leftmouth_y);
        fprintf(f, "\t\t'confidence': %f\n", box.score);
        fprintf(f, "\t}");
        if (i != size - 1)
            fprintf(f, ",\n");
        else
            fprintf(f, "\n");
    }
    fprintf(f, "]");
    fclose(f);
}

MtcnnLoader::MtcnnLoader(const char *img, const char *label, string net_name) {
    img_ptr = fopen(img, "rb");
    label_ptr = fopen(label, "rb");
    if (net_name == "pnet") {
        net_size = 12;
        label_step = (sizeof(int) + sizeof(float) * 4 + sizeof(float) * 10);
        image_step = (sizeof(unsigned char) * 3 * 12 * 12);
    } else if (net_name == "rnet") {
        net_size = 24;
        label_step = (sizeof(int) + sizeof(float) * 4 + sizeof(float) * 10);
        image_step = (sizeof(unsigned char) * 3 * 24 * 24);
    } else if (net_name == "onet") {
        net_size = 48;
        label_step = (sizeof(int) + sizeof(float) * 4 + sizeof(float) * 10);
        image_step = (sizeof(unsigned char) * 3 * 48 * 48);
    } else {
        printf("Error!\n");
        net_size = 0;
        label_step = 0;
        image_step = 0;
    }
}

void mtcnn_data_loader(const char *data_path, const char *label_path, vtensor &data_set, vtensor &label_set, int width, int size) {
    FILE *img = fopen(data_path, "rb");
    FILE *label = fopen(label_path, "rb");
    
    for (int i = 0; i < size; ++i) {
        int cls;
        float bbox[4];
        float landmark[10];
        fread(&cls, sizeof(int), 1, label);
        fread(bbox, sizeof(float), 4, label);
        fread(landmark, sizeof(float), 10, label);
        Tensor label_data;
        float *label_data_ptr = label_data.weight;
        *(label_data_ptr++) = ((float)cls);
        for (int i = 0; i < 4; ++i) {
            *(label_data_ptr++) = (bbox[i]);
        }
        for (int i = 0; i < 10; ++i) {
            *(label_data_ptr++) = (landmark[i]);
        }
        label_set.push_back(label_data);
        
        unsigned char pixel[3 * width * width];
        fread(pixel, sizeof(unsigned char), 3 * width * width, img);
        
        if (i < 0) {
            unsigned char pixel_R[width * width];
            unsigned char pixel_G[width * width];
            unsigned char pixel_B[width * width];
            for (int i = 0; i < width * width; ++i) {
                pixel_R[i] = pixel[i * 3  +  0];
                pixel_G[i] = pixel[i * 3  +  1];
                pixel_B[i] = pixel[i * 3  +  2];
            }
            
            string file_name = to_string(i);
            FILE *f = fopen(file_name.c_str(), "wb");
            fprintf(f, "P6\n%d %d\n255\n", width, width);
            fwrite(pixel, sizeof(unsigned char), 3 * width * width, f);
            fclose(f);
        }
        float normal_pixel[3 * width * width];
        int index = 0;
        for (int d = 0; d < 3; ++d) {
            for (int h = 0; h < width; ++h) {
                for (int w = 0; w < width; ++w) {
                    normal_pixel[index++] = ((float)pixel[((h * width) + w) * 3 + d] - 127.5) / 128.0;
                }
            }
        }
        data_set.push_back(Tensor(normal_pixel, width, width, 3));
    }
    fclose(img);
    fclose(label);
}

void mtcnn_evaluate(Neural_Network *nn, vtensor &data_set, vector<vfloat> &label_set) {
    int count = 0;
    int correct = 0;
    int pos = 0;
    int neg = 0;
    for (int i = 0; i < data_set.size(); ++i) {
        vfloat out = nn->Forward(&data_set[i])[0]->toVector();
        if (label_set[i][0] == 1) {
            if (out[1] > out[0]) {
                correct++;
                pos++;
            }
            count++;
        } else if (label_set[i][0] == 0) {
            if (out[0] > out[1]) {
                correct++;
                neg++;
            }
            count++;
        }
    }
    printf("Acc: %.2f%% pos: %d neg: %d count: %d\n", (float)correct / count * 100, pos, neg, count);
}

Tensor MtcnnLoader::getImg(int index) {
    fseek(img_ptr, index * image_step, SEEK_SET);
    unsigned char pixel[image_step];
    fread(pixel, 1, image_step, img_ptr);
    float normal_pixel[image_step];
    
    index = 0;
    for (int d = 0; d < 3; ++d) {
        for (int h = 0; h < net_size; ++h) {
            for (int w = 0; w < net_size; ++w) {
                normal_pixel[index++] = ((float)pixel[((h * net_size) + w) * 3 + d] - 127.5) / 128.0;
            }
        }
    }
    return Tensor(normal_pixel, net_size, net_size, 3);
}

int MtcnnLoader::getSize() {
    fseek(label_ptr, 0, SEEK_END);
    size_t size = ftell(label_ptr);
    int len = size & 0x7FFFFFFF;
    len /= 60;
    return len;
}

Tensor MtcnnLoader::getLabel(int index) {
    fseek(label_ptr, index * label_step, SEEK_SET);
    int cls;
    float bbox[4];
    float landmark[10];
    fread(&cls, sizeof(int), 1, label_ptr);
    fread(bbox, sizeof(float), 4, label_ptr);
    fread(landmark, sizeof(float), 10, label_ptr);
    Tensor label_data(1, 1, 15, 0);
    float *label_ptr = label_data.weight;
    *(label_ptr++) = ((float)cls);
    for (int i = 0; i < 4; ++i) {
        *(label_ptr++) = (bbox[i]);
    }
    for (int i = 0; i < 10; ++i) {
        *(label_ptr++) = (landmark[i]);
    }
    return label_data;
}

void MtcnnTrainer::train(int epoch, int decade_interval, float decade_rate) {
    auto rng = std::mt19937((unsigned)time(NULL));
    vector<int> index;
    float loss = 0;
    int data_set_size = loader->getSize();
    printf("Total %d data\n", data_set_size);
    for (int i = 0; i < data_set_size; ++i) {
        index.push_back(i);
    }
    int interval = data_set_size / 100;
    for (int i = 0; i < epoch; ++i) {
        printf("Epoch %d Training...\n", i + 1);
        loss = 0;
        shuffle(index.begin(), index.end(), rng);
        for (int j = 0; j < data_set_size; ++j) {
            Tensor img = loader->getImg(index[j]);
            Tensor label = loader->getLabel(index[j]);
            loss += trainer->train(img, label)[0];
            if (j % interval == 0) {
                printf("[%.2f%%] loss: %f\n", 100.0 * j / data_set_size, loss);
            }
        }
        if ((i + 1) % decade_interval == 0) {
            trainer->decade(decade_rate);
        }
        printf("Epoch %d Total loss: %f\n", i + 1, loss);
    }
}

void MtcnnTrainer::evaluate(Neural_Network &nn) {
    int count = 0;
    int correct = 0;
    int pos = 0;
    int neg = 0;
    int data_set_size = loader->getSize();
    int interval = data_set_size / 100;
    for (int i = 0; i < data_set_size; ++i) {
        Tensor img = loader->getImg(i);
        Tensor label = loader->getLabel(i);
        vfloat out = nn.Forward(&img)[0]->toVector();
        float *label_ptr = label.weight;
        if (label_ptr[0] == 1) {
            if (out[1] > out[0]) {
                correct++;
                pos++;
            }
            count++;
        } else if (label_ptr[0] == 0) {
            if (out[0] > out[1]) {
                correct++;
                neg++;
            }
            count++;
        }
        if (i % interval == 0) {
            printf("evaluating...%.2f%%\n", 100.0 * i / data_set_size);
        }
    }
    printf("Acc: %.2f%% pos: %d neg: %d count: %d\n", (float)correct / count * 100, pos, neg, count);
}
