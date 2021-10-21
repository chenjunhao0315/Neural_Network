//
//  Data_Process.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/4.
//

#ifndef Data_Process_hpp
#define Data_Process_hpp

#include <iostream>
#include <stdio.h>
#include "Tensor.hpp"
//#include "jpeg.h"

using std::string;

typedef vector<Tensor> vtensor;

class Data {
public:
    Data(string filename_, int width_, int height_, int dimension_, int offset_) : filename(filename_), width(width_), height(height_), dimension(dimension_), offset(offset_) {}
    vtensor get(int num);
private:
    string filename;
    int width;
    int height;
    int dimension;
    int offset;
    int num;
};


#endif /* Data_Process_hpp */
