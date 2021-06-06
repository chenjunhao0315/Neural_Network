//
//  Data_Process.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/4.
//


#include "Data_Process.hpp"

using std::string;

Data::Data(string filename_, int width_, int height_) {
    filename = filename_;
    width = width_;
    height = height_;
}

vtensor Data::get(int num) {
    FILE *f = fopen(filename.c_str(), "rb");
    vtensor data_set;
    data_set.resize(num);
    fseek(f, 80, SEEK_CUR);
    unsigned char c;
    Tensor data;
    for (int i = 0; i < num; i++) {
        vfloat img;
        for (int j = 0; j < width * height; ++j) {
            fread(&c, 1, 1, f);
            img.push_back((float)c / 255.0);
        }
        data = Tensor(img, img, img, 28, 28);
        data_set[i] = data;
        img.clear();
    }
    return data_set;
}
