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

using std::string;

typedef vector<Tensor> vtensor;

class Data {
public:
    Data(string filename_, int width_, int height_);
    vtensor get(int num);
private:
    string filename;
    int width;
    int height;
    int num;
};

#endif /* Data_Process_hpp */
