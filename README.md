# Neural_Network

## Introduction
This is a simple project to implement neural network in c++, the structure of network is inspired by [ConvNetJS][1], [Darknet][2] and [Caffe][4].

## Feature
* C++14
* Multi-thread support with Openmp
* Run only on CPU
* Structure visualization by [Netron][3] with Caffe2 like prototxt file
* Easy to add customed layer

## Supported Layers
* Input layer (data input)
* FullyConnected layer
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
Neural_Network nn("network_name");    // The default name is sequential
```
#### Add layers
It will add layer to neural network, checking the structure of input tensor at some layer, for instance, Concat layer, the input width and height should be the same at all input. **Note**: The **first** layer of network should be **Input layer**.
```cpp
nn.addLayer(LayerOption{{"type", "XXX"}, {"option", "YYY"}, {"input_name", "ZZZ"}, {"name", "WWW"}});    // The options are unordered
```
* Input layer options
> **input_width** <br>
> **input_height** <br>
> **input_dimension**
* FullyConnected layer options
> **number_neurons** <br>
> batchnorm (none) <br>
> activation (none)
* Convolution layer options
> **number_kernel** <br>
> **kernel_width** <br>
> kernel_height ( = kernel_width) <br>
> stride (1) <br>
> padding (0) <br>
> batchnorm (none) <br>
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
* Pooling layer (output_size = (input_size + padding - kernel_size) / stride + 1)
> **kernel_width** <br>
> kernel_height ( = kernel_width) <br>
> stride (1) <br>
> padding (0)
* AvgPooling layer
* UpSample layer
> **stride**
* ShortCut layer (single layer)
> **shortcut**
* Concat layer (multi layers)
> concat (none) <br>
> splits (1) <br>
> split_id (0)
* Softmax layer
* EuclideanLoss layer
* YOLOv3 layer
> **total_anchor_num** <br>
> **anchor_num** <br>
> **classes** <br>
> **max_boxes** <br>
> **anchor** <br>
> mask <br>
> ignore_iou_threshold (0.5) <br>
> truth_iou_threshold (1)
* YOLOv4 layer
> **total_anchor_num** <br>
> **anchor_num** <br>
> **classes** <br>
> **max_boxes** <br>
> **anchor** <br>
> mask <br>
> ignore_iou_threshold (0.5) <br>
> truth_iou_threshold (1) <br>
> scale_x_y (1) <br>
> iou_normalizer (0.75) <br>
> obj_normalizer (1) <br>
> cls_normalizer (1) <br>
> delta_normalizer (1) <br>
> beta_nms (0.6) <br>
> objectness_smooth (0) <br>
> label_smooth_eps (0) <br>
> max_delta (FLT_MAX) <br>
> iou_thresh (FLT_MAX) <br>
> new_coordinate (false) <br>
> focal_loss (false) <br>
> iou_loss (IOU) <br>
> iou_thresh_kind (IOU)

#### Add output
The output of network can be more than one, if you need, just type this command. The default output is the last layer of network.
```cpp
nn.addOutput("Layer_name");
```

#### Construct network
It will construct the computation graph of neural network automatically, but not checking is it reasonable or not.
```cpp
nn.compile(mini_batch_size);    // The default mini_batch_size is 1
```

#### Network shape
Show the breif detail between layer and layer.
```cpp
nn.shape();    // Show network shape
```

#### Visualization
It will output a Caffe2 like network topology file, but it is not a converter to convert the model to Caffe2.
```cpp
nn.to_prototxt("output_filename.prototxt");    // The default name is model.prootxt
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
nn.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "32"}, {"name", "connected"}, {"activation", "PRelu"}});
nn.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "3"}, {"name", "connected_2"}});
nn.addLayer(LayerOption{{"type", "Softmax"}, {"name", "softmax"}});
nn.compile();
nn.to_prototxt();
```
The graph shown by Nerton.
![image](https://github.com/chenjunhao0315/Neural_Network/blob/main/Example_Network.png)

#### Forward Propagation
The data flow of network is based on **Tensor**. To forward propagation, just past the pointer of data to `network.Forward(POINTER_OF_DATA)` function. And it will return a **vector** of **Tensor pointer**, the **vtensorptr** is defined by `std::vecotr<Tensor*>`.
```cpp
Tensor data(28, 28, 3);
vtensorptr output = nn.Forward(&data);
```

#### Backward Propagation
To backward propagation, just past the pointer of data to `network.Backward(POINTER_OF_LABEL)` function. And it will return the **loss** with floating point type.
```cpp
Tensor label(1, 1, 3); label = {0, 1, 0};
float loss = nn.Backward(&label);
```

#### Save model
Ah, just save model.
```cpp
nn.save("model_name.bin");
```

#### Load model
Ah, just load model.
```cpp
Neural_Network nn;
nn.load("model_name.bin");
```

#### Add customed layer
You can add the customed layer like [Caffe][4]. If you want to save customed layer, you should add some code at `Neural_Network::save()`, `Layer::save()` and `Neural_Network::load()`, it is not so hard, I'm trying to revise the network data structure to let it work automatically.
```cpp
#include "Layer.hpp"
class CustomedLayer : public BaseLayer {
public:
    CustomedLayer(Layeroption opt);
    void Forward(bool train);    // For normal layer
    void Forward(Tensor *input);    // For input layer
    void Backward(Tensor *target);
private:
    ...
};
REGISTER_LAYER_CLASS(Customed);
```

#### Save otter model &hearts;
If you add some customed layer, remember to write the definition of layer prarmeter in `layer.txt`, the syntax is like below.
```cpp
Customed {
    REQUIRED TYPE PARAMETER_NAME // for required parameter
    OPTION TYPE PARAMETER_NAME DEFAULT_VALUE // for optional parameter
}
```
Then, you can save the model without revise any code. The otter file is easy to read and revise but it is sensitive to **syntax**, edit it carefully. The **otter** model syntax is like below.
```cpp
name: "model_name"
output: OUTPUT_LAYER_1_NAME    // optional, can more than one
output: OUTPUT_LAYER_2_NAME
LayerType {
    name: LAYER_NAME    // optional
    input_name: INPUT_LAYER_NAME    // optional
    Param {
        LAYER_PARAMETER: PARAMETER    // Look up the above layer option
    }
    batchnorm: BOOL    // optional
    activation: ACTIVATION_LAYER    //optional
}
LayerType {
    ...
}
...
```
Just type,
```cpp
nn.save_otter("model_name");
```

#### Load otter model
```cpp
Neural_Network nn;
nn.load_otter("model_name");
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
> ro (0.95) <br>
> eps (1e-6)
* Trainer:&#58;Method::ADAM
> ro (0.95) <br>
> eps (1e-6) <br>
> beta_1 (0.9) <br>
> beta_2 (0.999)

###### Learning rate policy
* CONSTANT
* STEP
> **step** <br>
> **scale**
* STEPS
> **steps** <br>
> **step_X** <br>
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
vtensor dataset;    // Add data with any way
vtensor labelset;    // Add label with any way
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
* Method 1 <br>
You will get a **Tensor t** with random value inside.
```cpp
Tensor t(WIDTH, HEIGHT, DIMENSION);
```
* Method 2 <br>
You will get a **Tensor t** with identical value **PARAMETER** inside.
```cpp
Tensor t(WIDTH, HEIGHT, DIMENSION, PARAMETER);
```
* Method 3 <br>
You will get a **Tensor t** with the same value as the **vector** you past in with **width** and **height** are equal to **1**.
```cpp
vfloat v{1, 2, 3};
Tensor t(v);
```
* Method 4 <br>
You will get a **Tensor t** with the same value as the **array** you past in.
```cpp
float f[3] = {1, 2, 3};
Tensor t(f, 1, 1, 3);    // t = [1, 2, 3]
```

#### Tensor extend
Allocate the memory of **delta_weight** in Tensor. To save memory, it will not be allocated in default.
```cpp
Tensor t(WIDTH, HEIGHT, DIMENSION);
t.extend();
```

#### Operation on Tensor
* = (Tensor) <br>
Deep copy from a to b, including extend.
```cpp
Tensor a(1, 1, 2, 1);    // a = [1, 1]
Tensor b = a;   // b = [1, 1]
```
* = (float) <br>
Set all value to input
```cpp
Tensor a(1, 1, 3, 0);    // a = [0, 0, 0]
a = 1;  // a = [1, 1, 1]
```
* = (initializer list) <br>
Set the previous element to initialzer list
```cpp
Tensor a(1, 1, 5, 3);    // a = [3, 3, 3, 3, 3]
a = {1, 2, 4};  // a = [1, 2, 4, 3, 3]
```
* [INDEX] <br>
Revise or take the value at INDEX.
```cpp
Tensor a(1, 1, 5, 3);    // a = [3, 3, 3, 3, 3]
a[2] = 0;   // a = [3, 3, 0, 3, 3]
float value = a[4]; // value = 3
```
* += (Tensor)
```cpp
Tensor a(1, 1, 2, 1);    // a = [1, 1]
Tensor b(1, 1, 2, 2);    // b = [2, 2]
a += b; // a = [3, 3] b = [2, 2]
```
* -= (Tensor)
```cpp
Tensor a(1, 1, 2, 1);    // a = [1, 1]
Tensor b(1, 1, 2, 2);    // b = [2, 2]
a -= b; // a = [-1, -1] b = [2, 2]
```
* \+ (Tensor)
```cpp
Tensor a(1, 1, 2, 1);    // a = [1, 1]
Tensor b(1, 1, 2, 2);    // b = [2, 2]
Tensor c = a + b;   // c = [3, 3]
```
* \- (Tensor)
```cpp
Tensor a(1, 1, 2, 1);   // a = [1, 1]
Tensor b(1, 1, 2, 2);   // b = [2, 2]
Tensor c = a - b;   // c = [-1,-1]
```
* << <br>
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
    nn.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "4"}, {"activation", "Relu"}});
    nn.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "2"}, {"activation", "Softmax"}});
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
[4]: https://github.com/BVLC/caffe


