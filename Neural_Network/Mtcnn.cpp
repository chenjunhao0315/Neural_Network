//
//  Mtcnn.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/7/19.
//

#include "Mtcnn.hpp"

PNet::PNet(const char *model_name) {
    min_face_size = 25;
    scale_factor = 0.709;
    threshold[0] = 0.97;
    pnet = Neural_Network("parallel");
    pnet.load(model_name);
}

PNet::PNet() {
    min_face_size = 25;
    scale_factor = 0.709;
    threshold[0] = 0.96;
    pnet = Neural_Network("parallel");
    pnet.addLayer(LayerOption{{"type", "Input"}, {"input_width", "12"}, {"input_height", "12"}, {"input_dimension", "3"}, {"name", "input"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "10"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_1"}});
    pnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_1"}});
    pnet.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}, {"name", "pool_1"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_2"}});
    pnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_2"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_3"}});
    pnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_3"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "2"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_4"}});
    //    pnet.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "2"}});
    pnet.addLayer(LayerOption{{"type", "Softmax"}, {"name", "cls_prob"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "4"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_5"}, {"input_name", "prelu_3"}});
    pnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "bbox_pred"}, {"input_name", "conv_5"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "10"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_6"}, {"input_name", "prelu_3"}});
    pnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "land_pred"}, {"input_name", "conv_6"}});
    pnet.addOutput("cls_prob");
    pnet.addOutput("bbox_pred");
    pnet.addOutput("land_pred");
    pnet.makeLayer();
    pnet.shape();
}

vector<Bbox> PNet::detect(IMG &img) {
    int net_size = 12;
    float current_scale = float(net_size) / min_face_size;
//    float current_scale = (float)1000 / max(img.width, img.height);
//    float current_scale = 1.0;
//    if (min(img.width, img.height) > 325) {
//        current_scale = 325.0 / min(img.height, img.width);
//    } else if (max(img.width, img.height) < 325) {
//        current_scale = 325.0 / max(img.height, img.width);
//    }
//    if (min(img.width, img.height) > 1000) {
//        current_scale = 1000.0 / min(img.height, img.width);
//    } else if (max(img.width, img.height) < 1000) {
//        current_scale = 1000.0 / max(img.height, img.width);
//    }
//    printf("scale: %.2f origin: %.2f\n", current_scale, float(net_size) / min_face_size);
    
    IMG img_resize = img.resize(Size(), current_scale, current_scale);
    int current_width = img_resize.width;
    int current_height = img_resize.height;
    vector<Bbox> bbox_list;
    
    while(min(current_width, current_height) > net_size) {
//        auto start = high_resolution_clock::now();
//        auto stop = high_resolution_clock::now();
//        auto duration = duration_cast<milliseconds>(stop - start);
        Feature_map map = this->predict(img_resize);
//        stop = high_resolution_clock::now();
//        duration = duration_cast<milliseconds>(stop - start);
//        printf("Predict time: %lldms\n", duration.count());
//        start = high_resolution_clock::now();
        vector<Bbox> list = generate_bbox(map, current_scale, threshold[0]);
//        stop = high_resolution_clock::now();
//        duration = duration_cast<milliseconds>(stop - start);
//        printf("generate box time: %lldms\n", duration.count());
//        start = high_resolution_clock::now();
        list = nms(list, 0.5);
//        stop = high_resolution_clock::now();
//        duration = duration_cast<milliseconds>(stop - start);
//        printf("nms time: %lldms\n", duration.count());
        bbox_list.insert(bbox_list.end(), list.begin(), list.end());
        
        current_scale *= scale_factor;
//        start = high_resolution_clock::now();
        img_resize = img.resize(Size(), current_scale, current_scale);
//        stop = high_resolution_clock::now();
//        duration = duration_cast<milliseconds>(stop - start);
//        printf("resize time: %lldms\n", duration.count());
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
    int img_size = img.width * img.height * img.channel;
    float *img_data = new float [img_size];
    for (int i = 0; i < img_size; ++i) {
        img_data[i] = ((float)rgb[i] - 127.5) / 128.0;
    }
    float *data = input.weight;
    
    y = -padding;
    for (h = 0; h < map_height; ++h, y += stride) {
        x = -padding;
        for (w = 0; w < map_width; ++w, x += stride) {
            for (f_h = 0; f_h < 12; ++f_h) {
                coordinate_h = y + f_h;
                for (f_w = 0; f_w < 12; ++f_w) {
                    coordinate_w = x + f_w;
                    for (f_d = 0; f_d < 3; ++f_d) {
                        if (coordinate_w >= 0 && coordinate_w < width && coordinate_h >= 0 && coordinate_h < height)
                            data[((12 * f_h) + f_w) * 3 + f_d] = img_data[((width * coordinate_h) + coordinate_w) * 3 + f_d];
                        else
                            data[((12 * f_h) + f_w) * 3 + f_d] = 0;
                    }
                }
            }
            map[h][w] = pnet.Forward(&input);
        }
    }
    delete [] img_data;
    return map;
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
        int h = Bbox_list[i].y2 - Bbox_list[i].y1;
        int w = Bbox_list[i].x2 - Bbox_list[i].x1;
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
        int width = bbox_c[i].x2 - bbox_c[i].x1;
        int height = bbox_c[i].y2 - bbox_c[i].y1;
        bbox_c[i].x1 += bbox_c[i].dx1 * width;
        bbox_c[i].y1 += bbox_c[i].dy1 * height;
        bbox_c[i].x2 += bbox_c[i].dx2 * width;
        bbox_c[i].y2 += bbox_c[i].dy2 * height;
    }
    return bbox_c;
}

vector<Bbox> RNet::detect(IMG &img, vector<Bbox> &pnet_bbox) {
    vector<Bbox> square_bbox = convert_to_square(pnet_bbox);
    vector<Bbox> rnet_bbox;
    size_t square_bbox_size = square_bbox.size();
    for (int i = 0; i < square_bbox_size; ++i) {
        IMG crop = img.crop(Rect(square_bbox[i].x1, square_bbox[i].y1, square_bbox[i].x2, square_bbox[i].y2));
        crop = crop.resize(Size(24, 24));
        unsigned char *pixel = crop.toPixelArray();
        float *pixel_c = new float [3 * 24 * 24];
        for (int i = 0; i < 3 * 24 * 24; ++i) {
            pixel_c[i] = ((float)pixel[i] - 127.5) / 128;
        }
        Tensor crop_img(pixel_c, 24, 24, 3);
        vfloat rnet_detect = rnet.Forward(&crop_img);
        if (rnet_detect[1] > threshold[0]) {
            Bbox rnet_detect_box(square_bbox[i].x1, square_bbox[i].y1, square_bbox[i].x2, square_bbox[i].y2, rnet_detect[1], rnet_detect[2], rnet_detect[3], rnet_detect[4], rnet_detect[5]);
            rnet_bbox.push_back(rnet_detect_box);
        }
    }
    rnet_bbox = nms(rnet_bbox, 0.6);
    rnet_bbox = calibrate_box(rnet_bbox);
    return rnet_bbox;
}

RNet::RNet() {
    threshold[0] = 0.9;
    rnet = Neural_Network("parallel");
    rnet.addLayer(LayerOption{{"type", "Input"}, {"input_width", "24"}, {"input_height", "24"}, {"input_dimension", "3"}, {"name", "input"}});
    rnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "28"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_1"}});
    rnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_1"}});
    rnet.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "3"}, {"stride", "2"}, {"name", "pool_1"}, {"padding", "1"}});
    rnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "48"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_2"}});
    rnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_2"}});
    rnet.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "3"}, {"stride", "2"}, {"name", "pool_2"}});
    rnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "2"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_3"}});
    rnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_3"}});
    rnet.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "128"}, {"name", "fc_1"}});
    rnet.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "2"}, {"name", "fc_cls"}});
    rnet.addLayer(LayerOption{{"type", "Softmax"}, {"name", "cls_prob"}});
    rnet.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "4"}, {"name", "fc_bbox"}, {"input_name", "fc_1"}});
    rnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "bbox_pred"}, {"input_name", "fc_bbox"}});
    rnet.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "10"}, {"name", "fc_land"}, {"input_name", "fc_1"}});
    rnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "land_pred"}, {"input_name", "fc_land"}});
    rnet.addOutput("cls_prob");
    rnet.addOutput("bbox_pred");
    rnet.addOutput("land_pred");
    rnet.makeLayer();
    rnet.shape();
}

RNet::RNet(const char *model_name) {
    rnet = Neural_Network("parallel");
    rnet.load(model_name);
}

ONet::ONet() {
    threshold[0] = 0.9;
    onet = Neural_Network("parallel");
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
    onet.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "256"}, {"name", "fc_1"}});
    onet.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "2"}, {"name", "fc_cls"}});
    onet.addLayer(LayerOption{{"type", "Softmax"}, {"name", "cls_prob"}});
    onet.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "4"}, {"name", "fc_bbox"}, {"input_name", "fc_1"}});
    onet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "bbox_pred"}, {"input_name", "fc_bbox"}});
    onet.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "10"}, {"name", "fc_land"}, {"input_name", "fc_1"}});
    onet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "land_pred"}, {"input_name", "fc_land"}});
    onet.addOutput("cls_prob");
    onet.addOutput("bbox_pred");
    onet.addOutput("land_pred");
    onet.makeLayer();
    onet.shape();
}

ONet::ONet(const char *model_name) {
    onet = Neural_Network("parallel");
    onet.load(model_name);
}

vector<Bbox> ONet::detect(IMG &img, vector<Bbox> &rnet_bbox) {
    vector<Bbox> square_bbox = convert_to_square(rnet_bbox);
    vector<Bbox> onet_bbox;
    size_t square_bbox_size = square_bbox.size();
    for (int i = 0; i < square_bbox_size; ++i) {
        IMG crop = img.crop(Rect(square_bbox[i].x1, square_bbox[i].y1, square_bbox[i].x2, square_bbox[i].y2));
        crop = crop.resize(Size(48, 48));
        unsigned char *pixel = crop.toPixelArray();
        Tensor crop_img(48, 48, 3);
        float *pixel_c = crop_img.weight;
        for (int i = 0; i < 3 * 48 * 48; ++i) {
            pixel_c[i] = ((float)pixel[i] - 127.5) / 128;
        }
        
        vfloat onet_detect = onet.Forward(&crop_img);
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
    onet_bbox = nms(onet_bbox, 0.6, 1);
    
    return onet_bbox;
}

void mtcnn_data_loader(const char *data_path, const char *label_path, vtensor &data_set, vector<vfloat> &label_set, int width, int size) {
    FILE *img = fopen(data_path, "rb");
    FILE *label = fopen(label_path, "rb");
    
    for (int i = 0; i < size; ++i) {
        int cls;
        float bbox[4];
        float landmark[10];
        fread(&cls, sizeof(int), 1, label);
        fread(bbox, sizeof(float), 4, label);
        fread(landmark, sizeof(float), 10, label);
        vfloat label_data;
        label_data.push_back((float)cls);
        for (int i = 0; i < 4; ++i) {
            label_data.push_back(bbox[i]);
        }
        for (int i = 0; i < 10; ++i) {
            label_data.push_back(landmark[i]);
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
        for (int i = 0; i < 3 * width * width; ++i) {
            normal_pixel[i] = ((float)pixel[i] - 127.5) / 128.0;
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
        vfloat out = nn->Forward(&data_set[i]);
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

Mtcnn::~Mtcnn() {
    delete pnet;
    delete rnet;
}

Mtcnn::Mtcnn(const char *model_pnet, const char *model_rnet, const char *model_onet) {
    pnet = new PNet(model_pnet);
    rnet = new RNet(model_rnet);
    pnet->min_face_size = 50;
    pnet->threshold[0] = 0.93;
    rnet->threshold[0] = 0.7;
}

vector<Bbox> Mtcnn::detect(IMG &img) {
    
    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    vector<Bbox> bbox = pnet->detect(img);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    printf("PNet Get %d proposal box! time: %lldms\n", (int)bbox.size(), duration.count());

    IMG pnet_detect(img);
    for (int i = 0; i < bbox.size(); ++i) {
        pnet_detect.drawRectangle(Rect{(bbox[i].x1), (bbox[i].y1), (bbox[i].x2), (bbox[i].y2)}, RED);
    }
    pnet_detect.save("pnet_predict.jpg", 80);

    start = high_resolution_clock::now();
    vector<Bbox> rnet_bbox = rnet->detect(img, bbox);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start); 
    printf("RNet Get %d proposal box! time: %lldms\n", (int)rnet_bbox.size(), duration.count());

    IMG rnet_detect(img);
    for (int i = 0; i < rnet_bbox.size(); ++i) {
        rnet_detect.drawRectangle(Rect{(rnet_bbox[i].x1), (rnet_bbox[i].y1), (rnet_bbox[i].x2), (rnet_bbox[i].y2)}, Color(255, 0, 0));
    }
    rnet_detect.save("rnet_predict.jpg", 80);
    return rnet_bbox;
}
