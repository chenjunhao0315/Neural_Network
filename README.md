# Neural_Network

## Introduction
This is a simple project to implement neural network in c++, the structure of network is inspired by [ConvNetJS][1] and [Darknet][2].

## Feature
* C++14
* Multi-thread support with Openmp
* Run only on CPU
* Structure visualization by [Netron][3] with Caffee2 like prototxt file

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

## Supported Trainers
* SGD
* ADADELTA
* ADAM

## Construct network
#### Initialize network 
Declear the nerual network. If the backpropagated path is special, you should name it and define the backpropagated path in `Neural_Network::Backward()` at `Neural_Network.cpp`.
```cpp
Neural_Network nn("network_name");	// The default name is sequential
```
#### Add layers
It will add layer to neural network, checking the structure of input tensor at some layer, for instance, Concat layer, the input width and height should be the same at all input. **Note**: The **first** layer of network should be **Input layer**.
```cpp
nn.addLayer(LayerOption{{"type", "XXX"}, {"option", "YYY"}, {"input_name", "ZZZ"}, {"name", "WWW"}});	// The options are unordered
```
* Input layer options
> **input_width**
> **input_height**
> **input_dimension**
* Fullyconnected layer options
> **number_neurons**
> batchnorm (none)
> activation (none)
* Convolution layer options
> **number_kernel**
> **kernel_width**
> kernel_height ( = kernel_width)
> stride (1)
> padding (0)
> batchnorm (none)
> actiivation (none)
* BatchNormalization layer
* Relu layer
* PRelu layer
> alpha (0.25)
* LRelu layer
> alpha (0.1)
* Sigmoid layer
* Mish layer
* Swish layer
* Dropout layer
> probability (0.5)
* Pooling layer (output = (layer_size + padding - kernel_size) / stride + 1)
> **kernel_width**
> kernel_height ( = kernel_width)
> stride (1)
> padding (0)
* AvgPooling layer
* UpSample layer
> **stride**
* ShortCut layer (single layer)
> **shortcut**
* Concat layer (multi layers)
> concat (none)
> splits (1)
> split_id (0)
* Softmax layer
* EuclideanLoss layer
* Yolov3 layer
> **total_anchor_num**
> **anchor_num**
> **classes**
> **max_boxes**
> **anchor**
> mask
> ignore_iou_threshold (0.5)
> truth_iou_threshold (1)
* Yolov4 layer
> **total_anchor_num**
> **anchor_num**
> **classes**
> **max_boxes**
> **anchor**
> mask
> ignore_iou_threshold (0.5)
> truth_iou_threshold (1)
> scale_x_y (1)
> iou_normalizer (0.75)
> obj_normalizer (1)
> cls_normalizer (1)
> delta_normalizer (1)
> beta_nms (0.6)
> objectness_smooth (0)
> label_smooth_eps (0)
> max_delta (FLT_MAX)
> iou_thresh (FLT_MAX)
> new_coordinate (false)
> focal_loss (false)
> iou_loss (IOU)
> iou_thresh_kind (IOU)

#### Add output
The output of network can be more than one, if you need, just type this command. The default output is the last layer of network.
```cpp
nn.addOutput("Layer_name");
```

#### Construct network
It will construct the computation graph of neural network automatically, but not checking is it reasonable or not.
```cpp
nn.compile(mini_batch_size);	// The default mini_batch_size is 1
```

#### Network shape
Show the breif detail between layer and layer.
```cpp
nn.shape();	// Show network shape
```

#### Visualization
It will output a Caffee2 like network topology file, but it is not a converter to convert the model to Caffee2.
```cpp
nn.to_prototxt("output_filename.prototxt");	// The default name is model.prootxt
```
Open the file at [Netron][3] to see the network structure.

#### Example of constructing a network
The example is a classifier with 3 classes, and the input is 28x28x3 tensor, it works may not be so well, just try to demonstrate all kinds of layer.
```cpp
Neural_Network nn;
nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "28"}, {"input_height", "28"}, {"input_dimension", "3"}, {"name", "input"}});
nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "Relu"}, {"name", "conv_1"}});
nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "LRelu"}, {"name", "conv_2"}});
nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "8"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "LRelu"}, {"name", "conv_3"}});
nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "LRelu"}, {"name", "conv_4"}});
nn.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "lr_conv_2"}, {"name", "shortcut_1"}});
nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "8"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "Mish"}, {"name", "conv_5"}});
nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "8"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "Mish"}, {"name", "conv_6"}, {"input_name", "re_conv_1"}});
nn.addLayer(LayerOption{{"type", "Concat"}, {"concat", "mi_conv_5"}, {"splits", "1"}, {"split_id", "0"}, {"name", "concat"}});
nn.addLayer(LayerOption{{"type", "Concat"}, {"splits", "2"}, {"split_id", "1"}, {"name", "concat_4"}});
nn.addLayer(LayerOption{{"type", "Dropout"}, {"probability", "0.2"}, {"name", "dropout"}});
nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "Swish"}, {"name", "conv_7"}, {"input_name", "concat"}});
nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}, {"name", "pool_1"}, {"input_name", "concat"}});
nn.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sw_conv_7"}, {"name", "shortcut_2"}});
nn.addLayer(LayerOption{{"type", "UpSample"}, {"stride", "2"}, {"name", "upsample"}});
nn.addLayer(LayerOption{{"type", "Concat"}, {"concat", "dropout"}, {"splits", "1"}, {"split_id", "0"}, {"name", "concat_3"}});
nn.addLayer(LayerOption{{"type", "AvgPooling"}, {"name", "avg_pooling"}, {"input_name", "concat"}});
nn.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "avg_pooling"}, {"name", "shortcut_3"}, {"input_name", "concat_3"}});
nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "32"}, {"name", "connected"}, {"activation", "PRelu"}});
nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "3"}, {"name", "connected_2"}});
nn.addLayer(LayerOption{{"type", "Softmax"}, {"name", "softmax"}});
nn.compile();
nn.to_prototxt();
```
The graph shown by Nerton.
![image](https://github.com/chenjunhao0315/Neural_Network/blob/main/Example_Network.png)

#### Forward Propagation
The data flow of network is based on **Tensor**. To forward propagation, just past the pointer of data to `network.Forward(POINTER_OF_DATA)` functioin. And it will return a **vector** of **Tensor pointer**, the **vtensorptr** is defined by `std::vecotr<Tensor*>`.
```cpp
Tensor data(28, 28, 3);
vtensorptr output = nn.Forward(&data);
```

#### Backward Propagation
To backward propagation, just past the pointer of data to `network.Backward(POINTER_OF_LABEL)` functioin. And it will return the **loss** with floating point type.
```cpp
Tensor label(1, 1, 3); label = {0, 1, 0};
float loss = nn.Backward(&label);
```

#### Get training arguments
If you want to train with your own method, you can use this command to get the whole **weights** and **delta weights** in network, if the layer is trainable. It will return a **vector** of **Train_Args**. The **Train_Args** is defined by,
```cpp
struct Train_Args {
    bool valid;
    Tensor *kernel;
    Tensor *biases;
    int kernel_size;
    int kernel_list_size;
    int biases_size;
    vfloat ln_decay_list;
};
```
```cpp
vector<Train_Args> args_list = network.getTrainArgs();
```
You can update the weight by your own method, or just use the **Trainer** below to update the network weight automatically.

## Construct trainer
#### Initialze the trainer
```cpp
Trainer trainer(&network, TrainerOption{{"method", XXX}, {"trainer_option", YYY}, {"policy", ZZZ}, {"learning_rate", WWW}, {"sub_division", TTT});
```

###### Trainer options
* Trainer:&#58;Method::SGD
> momentum (0.9)
* Trainer:&#58;Method::ADADELTA
> ro (0.95)
> eps (1e-6)
* Trainer:&#58;Method::ADAM
> ro (0.95)
> eps (1e-6)
> beta_1 (0.9)
> beta_2 (0.999)

###### Learning rate policy
* CONSTANT
* STEP
> **step**
> **scale**
* STEPS
> **steps**
> **step_X**
> **scale_X**
* EXP
> gamma (1)
* POLY
> power (4)
* RANDOM
> power(4)
* SIG
> gamma (1)

###### Warmup
* **warmup**

#### Start training
There are two method for training.
* Method 1
Put all data and label into **two vector of Tensor**, and past to the function `trainer.train_batch(DATA_SET, LABEL_SET, EPOCH)`, it will do everything automatically, such as shuffle the data, batch composition, etc. The **vtensor** is define by `std::vector<Tensor>`.
```cpp
vtensor dataset;	// Add data with any way
vtensor labelset;	// Add label with any way
trainer.train_batch(dataset, labelset, EPOCH);
```
* Method 2
Train the network with single data. Careful to the **data** and **label**, it should be extended with your **mini_batch_size** of network.
```cpp
Tensor data;
Tensor label;
trainer.train_batch(data, label);
```

## Tensor
It is the data structure used in this neural networkm, data arrangement is NCHW. The **Tensor** class is defined by,
```cpp
class Tensor {
    int width;
    int height;
    int dimension;
    int size;
    float* weight;
    float* delta_weight;
};
```

#### Initialize the tensor
* Method 1
You will get a **Tensor t** with random value inside.
```cpp
Tensor t(WIDTH, HEIGHT, DIMENSION);
```
* Method 2
You will get a **Tensor t** with identical value **PARAMETER** inside.
```cpp
Tensor t(WIDTH, HEIGHT, DIMENSION, PARAMETER);
```
* Method 3
You will get a **Tensor t** with the same value as the **vector** you past in with **width** and **height** are equal to **1**.
```cpp
vfloat v{1, 2, 3};
Tensor t(v);
```
* Method 4
You will get a **Tensor t** with the same value as the **array** you past in.
```cpp
float f[3] = {1, 2, 3};
Tensor t(f, 1, 1, 3);	// t = [1, 2, 3]
```

#### Tensor extend
Allocate the memory of **delta_weight** in Tensor. To save memory, it will not be allocated in default.
```cpp
Tensor t(WIDTH, HEIGHT, DIMENSION);
t.extend();
```

#### Operation on Tensor
* = (Tensor)
Deep copy from a to b, including extend.
```cpp
Tensor a(1, 1, 2, 1);	// a = [1, 1]
Tensor b = a;   // b = [1, 1]
```
* = (float)
Set all value to input
```cpp
Tensor a(1, 1, 3, 0);	// a = [0, 0, 0]
a = 1;  // a = [1, 1, 1]
```
* = (initializer list)
Set the previous element to initialzer list
```cpp
Tensor a(1, 1, 5, 3);	// a = [3, 3, 3, 3, 3]
a = {1, 2, 4};  // a = [1, 2, 4, 3, 3]
```
* [INDEX]
Revise or take the value at INDEX.
```cpp
Tensor a(1, 1, 5, 3);	// a = [3, 3, 3, 3, 3]
a[2] = 0;   // a = [3, 3, 0, 3, 3]
float value = a[4]; // value = 4
```
* += (Tensor)
```cpp
Tensor a(1, 1, 2, 1);	// a = [1, 1]
Tensor b(1, 1, 2, 2);	// b = [2, 2]
a += b; // a = [3, 3] b = [2, 2]
```
* -= (Tensor)
```cpp
Tensor a(1, 1, 2, 1);	// a = [1, 1]
Tensor b(1, 1, 2, 2);	// b = [2, 2]
a -= b; // a = [-1, -1] b = [2, 2]
```
* \+ (Tensor)
```cpp
Tensor a(1, 1, 2, 1);	// a = [1, 1]
Tensor b(1, 1, 2, 2);	// b = [2, 2]
Tensor c = a + b;   // c = [3, 3]
```
* \- (Tensor)
```cpp
Tensor a(1, 1, 2, 1);   // a = [1, 1]
Tensor b(1, 1, 2, 2);   // b = [2, 2]
Tensor c = a - b;   // c = [-1,-1]
```
* <<
Print all **weighs** in Tensor.

## Build and run
If the project doesn't include YOLOv4 layer, you can build with
* `$ g++ -Ofast -fopenmp -o nn *.cpp`

Else, the isnan() function is not working with -Ofast flag
* `$ g++ -O3 -fopenmp -o nn *.cpp`

Run with
* `$ ./nn`

## Example
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
