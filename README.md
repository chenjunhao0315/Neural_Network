# Neural_Network

# Introduction
This is a simple project to implement neural network in c++. The structure of network is inspired by [ConvNetJS][1] and [Darknet][2].

# Feature
* C++14
* Multi-thread support with Openmp
* Run only on CPU
* Structure visualization with [Netron][3] by Caffee2 like prototxt file

## Supported Layers
* Input layer (data input)
* Fullyconnected layer
* Convolution layer
* BatchNormalization layer
* Relu layer
* PRelu layer
* LRelu layer
* Sigmoid layer
* Mish layer
* Swish layer
* Dropout layer
* Pooling layer
* AvgPooling layer
* UpSample layer
* ShortCut layer (single layer)
* Concat layer (multi layers)
* Softmax layer
* EuclideanLoss layer
* Yolov3 layer
* Yolov4 layer

# Build and run
If the project doesn't include YOLOv4 layer, you can build with
* `$ g++ -Ofast -fopenmp -o nn *.cpp`

Else, the isnan() function is not working with -Ofast flag
* `$ g++ -O3 -fopenmp -o nn *.cpp`

Run with
* `$ ./nn`

# Example
The XOR problem example
```c++
#include <iostream>

#include "Neural_Network.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    Neural_Network nn;
    nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "1"}, {"input_height", "1"}, {"input_dimension", "2"}, {"name", "data"}});
    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "4"}, {"activation", "Relu"}});
    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "2"}, {"activation", "Softmax"}});
    nn.compile();

    Tensor a(1, 1, 2, 0);
    Tensor b(1, 1, 2, 1);
    Tensor c(1, 1, 2); c = {0, 1};
    Tensor d(1, 1, 2); d = {1, 0};
    vtensor data{a, b, c, d};
    Tensor a_l(1, 1, 1, 0);
    Tensor b_l(1, 1, 1, 0);
    Tensor c_l(1, 1, 1, 1);
    Tensor d_l(1, 1, 1, 1);
    vtensor label{a_l, b_l, c_l, d_l};

    Trainer trainer(&nn, TrainerOption{{"method", Trainer::Method::SGD}, {"learning_rate", 0.1}, {"warmup", 5}});
    trainer.train_batch(data, label, 100);
    
    printf("Input (0, 0) -> %.0f\n", nn.predict(&a)[0]);
    printf("Input (0, 1) -> %.0f\n", nn.predict(&c)[0]);
    printf("Input (1, 0) -> %.0f\n", nn.predict(&d)[0]);
    printf("Input (1, 1) -> %.0f\n", nn.predict(&b)[0]);
    
    return 0;
}
```

[1]: https://cs.stanford.edu/people/karpathy/convnetjs/
[2]: https://github.com/pjreddie/darknet
[3]: https://netron.app
