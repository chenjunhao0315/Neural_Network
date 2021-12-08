# Neural_Network

## About
This is a simple project to implement neural network in c++, the structure of network is inspired by [ConvNetJS][1], [Darknet][2], [Caffe][4] and [PyTorch][9].

The main motivation for me to write this framework is the homework from NTHU EE231002 Lab14 Image Processing. There is a task need to add a box around my head and I want it to do automatically, when I discover this paper, [MTCNN][6], I know it is time for me to construct a neural network from scratch.

First, I found this on youtube, [10.1: Introduction to Neural Networks - The Nature of Code][8], I learn some knowledge about neural network, but javascript? I think for a while, my poor coding ability and poor algorithm may make the whole project unusable or take uncountable time to run, so i turn to c++. (Maybe it is my bias that I think c/c++ is more efficient than javascript) At early stage, I take a reference from [ConvNetJS][1], the code is easy to understand, even until now, the **trainer** of this project is mostly indentical to that.

After finishing MTCNN, aiming to construct more complex network, I look for some object detection model, but most of them are based on Tensorflow or PyTorch, if I have any problem during implementation, it is hard to trace code to check if I am right or wrong. Then, I found [YOLO][7], it is based on [Darknet][2], a framework that not so complex, human readiable, give me a chance to build more complex network by myself. There are lots of layers in this project based on Darknet but maybe easier to read than the original version.

At the time I finish YOLO, I find that my layer management is too complex due to bad data structure. I noticed [Caffe][4]'s layer registry by accident, that is what I want. Revise the code right away to manage my layer like that, creating new format to define network structure called **otter** to store network topology replacing the old tedious method.

SSD...<br>
RNN...<br>
FASTER_RCNN...<br>

Introduce new Tensor data structure based on libtorch, try to implement autograd system with static and dynamic graph.

## Feature
* C++11
* No dependencies
* Multi-thread support with OpenMp
* Run only on CPU
* Structure visualization by [Netron][3] with Caffe2 like prototxt file
* Easy to add custom layer
* Python interface

## Supported Layers
#### Data layer
* Input layer (raw data input)
* Data layer (support data transform)

#### Vision layers
* Convolution layer (depthwise support)
* Pooling layer (maxpooling)
* AvgPooling layer
* UpSample layer

#### Common layers
* Dropout layer
* FullyConnected layer

#### Activation layers
* Sigmoid layer
* Tanh layer
* Relu layer
* PRelu layer
* LRelu layer
* Mish layer
* Swish layer
* Elu layer

#### Normalization layer
* BatchNormalization layer

#### Utility layers
* Concat layer (multi layers)
* Eltwise layer (multi layers)
* ShortCut layer (single layer)
* ScaleChannel layer

#### Loss layers
* Softmax layer (with cross entropy loss)
* EuclideanLoss layer

#### Special layers
* Yolov3 layer
* Yolov4 layer

## Supported Trainers
* SGD
* ADADELTA
* ADAM

## Supported Model
* MTCNN
* YOLOv3
* YOLOv3-tiny
* YOLOv3-openimage
* YOLOv4
* YOLOv4-tiny
* YOLOv4-csp

## Construct network
#### Initialize network 
Declear the nerual network. If the backpropagated path is special, you should name it and define the backpropagated path in `Neural_Network::Backward()` at `Neural_Network.cpp`.
```cpp
Neural_Network nn("network_name");    // The default name is sequential
```
#### Add layers
It will add layer to neural network, checking the structure of input tensor at some layer, for instance, Concat layer, the input width and height should be the same as all input. **Note**: The **first** layer of network should be **Input layer** or custom **Data layer**.
```cpp
nn.addLayer(LayerOption{{"type", "XXX"}, {"option", "YYY"}, {"input_name", "ZZZ"}, {"name", "WWW"}});    // The options are unordered
```
##### Data layer
* Input layer options
> **input_width** <br>
> **input_height** <br>
> **input_dimension**
* Data layer options
> scale (1) <br>
> mean (none) usage: "mean_1, mean_2, ...", same dimension as input

##### Vision layers
* Convolution layer options
> **number_kernel** <br>
> **kernel_width** <br>
> kernel_height ( = kernel_width) <br>
> stride (1) <br>
> stride_x (-1) <br>
> stride_y (-1) <br>
> dilation (1) <br>
> padding (0) <br>
> groups (1) <br>
> batchnorm (none) <br>
> activation (none)
* Pooling layer (output_size = (input_size + padding - kernel_size) / stride + 1)
> **kernel_width** <br>
> kernel_height ( = kernel_width) <br>
> stride (1) <br>
> padding (0)
* AvgPooling layer
* UpSample layer
> **stride**

##### Common layers
* Dropout layer
> probability (0.5)
* FullyConnected layer options
> **number_neurons** <br>
> batchnorm (none) <br>
> activation (none)

##### Activation layers
* Sigmoid layer
* Tanh layer
* Relu layer
* PRelu layer
> alpha (0.25)
* LRelu layer
> alpha (1)
* Mish layer
* Swish layer
* Elu layer
> alpha (0.1)

##### Normalization layer
* BatchNormalization layer

##### Utility layers
* Concat layer (multi layers)
> concat (none) <br>
> splits (1) <br>
> split_id (0)
* Eltwise layer
> **eltwise** <br>
> eltwise_op (prod, sum, max)
* ShortCut layer (single layer)
> **shortcut** <br>
> alpha (1) <br>
> beta (1)
* ScaleChannel layer
> **scalechannel**

##### Loss layers
* Softmax layer
* EuclideanLoss layer

##### Special layers
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
It will construct the static computation graph of neural network automatically, but not checking is it reasonable or not.
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
The data flow of network is based on **Tensor**. To forward propagation, just past the pointer of data to `network.Forward(POINTER_OF_DATA)` function. And it will return a **pointer** to **Tensor pointer**. Careful to use the output, it is the direct result of Neural Network!
```cpp
Tensor data(1, 3, 28, 28);
Tensor** output = nn.Forward(&data);
```

#### Backward Propagation
To backward propagation, just past the pointer of data to `network.Backward(POINTER_OF_LABEL)` function. And it will return the **loss** with floating point type.
```cpp
Tensor label(1, 1, 1, 3); label = {0, 1, 0};
float loss = nn.Backward(&label);
```

#### Extract Temporary Result
If you want to extract some result from the inner layer,
```cpp
Tensor temp;
nn.extract("LAYERNAME", temp);
```

#### Add custom layer
You can add the custom layer like [Caffe][5]. Save model as **otter** model like below! If defined correctly, it will save everything automatically.
```cpp
#include "Layer.hpp"
class CustomLayer : public BaseLayer {
public:
    CustomLayer(Layeroption opt);
    void Forward(bool train);    // For normal layer
    void Forward(Tensor *input);    // For input layer
    void Backward(Tensor *target);    // The output tensor should be extended! Or it will cause segmentation fault when clearing gradient (maybe fix at next version)
    vtensorptr connectGraph(vtensorptr input_tensor_, float *workspace);    // If there are multi inputs or need workspace
    inline const char* Type() const {return "Custom_Name";}
private:
    ...
};
REGISTER_LAYER_CLASS(Custom);
```
Remeber to add enum at `Layer.hpp`.

In the constrctor of custom layer, you can use ask space for storing data, for example
```cpp
CustomLayer::CustomLayer(Layeroption opt) : BaseLayer(opt) {
    this->applyInput(NUM);    // ask for input space to store input tensor (default = 1) Note: Data layer should set it to 0
    this->applyOutput(NUM);    // ask for output space to store output tensor (default = 1)
    this->applyKernel(NUM);    // ask for kernel space to store data
    kernel[0] = Tensor(BATCH, CHANNEL, HEIGHTWIDTH, PARAMETER);
    kernel[0].extend();    // If the kernel parameter can be updated
    kernel[1] = ...
    this->applyBias(NUM);    // ask for biases space to store data
    biases[0] = Tensor(BATCH, CHANNEL, HEIGHTWIDTH, PARAMETER);
    biases[0].extend();
    biases[1] = ...
}
```
If the layer can be trained, you need to pass train arguments to trainer, you need to add code at `BaseLayer::getTrainArgs()`, return the traing arguments, the traing arguments is defined by, 
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

#### Get training arguments
If you want to train with your own method, you can use this command to get the whole **weights** and **delta weights** in network, if the layer is trainable. It will return a **vector** of **Train_Args**.
```cpp
vector<Train_Args> args_list = network.getTrainArgs();
```
You can update the weight by your own method, or just use the **Trainer** below to update the network weight automatically.

#### Save otter model &hearts;
If you add some custom layer, remember to write the definition of layer prarmeter in `layer.txt`, the syntax is like below.
```cpp
Customed {
    REQUIRED TYPE PARAMETER_NAME // for required parameter (three parameters with two spaces)
    OPTION TYPE PARAMETER_NAME DEFAULT_VALUE // for optional parameter (four parameters with three spaces)
    OPTION multi/single connect PARAMETER    // If layer need extra input
    REQUIRED int net    // If layer need network input size
}
```
Then, you can save the model without revise any code. The otter file is easy to read and revise but it is sensitive to **syntax**, edit it carefully. The **otter** model syntax is like below.
```cpp
name: "model_name"
output: OUTPUT_LAYER_1_NAME    // optional, can more than one
output: OUTPUT_LAYER_2_NAME
# you can write comment in one line after hash mark
LayerType {
    name: LAYER_NAME    // optional
    input_name: INPUT_LAYER_NAME    // optional
    Param {
        LAYER_PARAMETER: PARAMETER    // Look up the above layer option
        LAYER_PARAMETER: "PARAMETER"    // If the parameter contain space, remember to add quotation mark
    }
    batchnorm: BOOL    // optional
    activation: ACTIVATION_LAYER    //optional
}
LayerType {
    ...
}
...
```
Just type this command, it will generate one or two file `mode_name.otter`, `model_name.dam`, first is the network structure file, second is the network weights file.
```cpp
nn.save_otter("model_name.otter", BOOL);    // true for saving .dam file
```
Or you can just save the network weights, by typing this command,
```cpp
nn.save_dam("weights_name.dam");
```
**New!!** Save network structure and weights into one file!!
```cpp
nn.save_ottermodel("model_name.ottermodel");
```

#### Load otter model
You can load the model with different way, structure only`.otter`file, weights only`.dam`file, or structure with weights`.ottermodel`file.
```cpp
Neural_Network nn;
nn.load_otter("model_name.otter", BATCH_SIZE);    
```
Or just load the weights, by typing this command,
```cpp
nn.load_dam("weight_name.dam");
```
**New!!** Load network structure and weights from one file!!
```cpp
nn.load_ottermodel("model_name.ottermodel", BATCH_SIZE);
```

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
    int batch;
    int channel;
    int height;
    int size;
    float* weight;
    float* delta_weight;
};
```

#### Initialize the tensor
* Method 1 <br>
You will get a **Tensor t** with random value inside.
```cpp
Tensor t(BATCH, CHANNEL, HEIGHT, WIDTH);
```
* Method 2 <br>
You will get a **Tensor t** with identical value **PARAMETER** inside.
```cpp
Tensor t(BATCH, CHANNEL, HEIGHT, WIDTH, PARAMETER);
```
* Method 3 <br>
You will get a **Tensor t** with the same value as the **vector** you past in with **batch**, **height** and **width** are equal to **1**.
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
Tensor t(BATCH, CHANNEL, HEIGHT, WIDTH);
t.extend();
```

#### Tensor reshape
Reshape the Tensor and clear all data.
```cpp
Tensor t(BATCH, CHANNEL, HEIGHT, WIDTH);
t.reshape(BATCH, CHANNEL, HEIGHT, WIDTH, EXTEND);
```

#### Operation on Tensor
* = (Tensor) <br>
Deep copy from a to b, including extend.
```cpp
Tensor a(1, 1, 1, 2, 1);    // a = [1, 1]
Tensor b = a;   // b = [1, 1]
```
* = (float) <br>
Set all value as input
```cpp
Tensor a(1, 1, 1, 3, 0);    // a = [0, 0, 0]
a = 1;  // a = [1, 1, 1]
```
* = (initializer list) <br>
Set the previous elements as initialzer list
```cpp
Tensor a(1, 1, 1, 5, 3);    // a = [3, 3, 3, 3, 3]
a = {1, 2, 4};  // a = [1, 2, 4, 3, 3]
```
* [INDEX] <br>
Revise or take the value at INDEX.
```cpp
Tensor a(1, 1, 1, 5, 3);    // a = [3, 3, 3, 3, 3]
a[2] = 0;   // a = [3, 3, 0, 3, 3]
float value = a[4]; // value = 3
```
* += (Tensor)
```cpp
Tensor a(1, 1, 1, 2, 1);    // a = [1, 1]
Tensor b(1, 1, 1, 2, 2);    // b = [2, 2]
a += b; // a = [3, 3] b = [2, 2]
```
* -= (Tensor)
```cpp
Tensor a(1, 1, 1, 2, 1);    // a = [1, 1]
Tensor b(1, 1, 1, 2, 2);    // b = [2, 2]
a -= b; // a = [-1, -1] b = [2, 2]
```
* \+ (Tensor)
```cpp
Tensor a(1, 1, 1, 2, 1);    // a = [1, 1]
Tensor b(1, 1, 1, 2, 2);    // b = [2, 2]
Tensor c = a + b;   // c = [3, 3]
```
* \- (Tensor)
```cpp
Tensor a(1, 1, 1, 2, 1);   // a = [1, 1]
Tensor b(1, 1, 1, 2, 2);   // b = [2, 2]
Tensor c = a - b;   // c = [-1,-1]
```
* << <br>
Print all **weights** in Tensor.

## Python Interface
### Neural_Network
Only inference mode now! The neural network is not completed with python interface, just for convenience used with some visualize UI, like matplotlib, etc. Before you use the python interface, you should build the library `otter.so` first!
#### Initialize network 
Initialize network,
```python
nn = Neural_Network(NETWORK_NAME)
```

#### Load ottermodel
Load ottermodel from file, 
```python
nn.load_ottermodel(MODEL_NAME)
```

#### Network shape
Show the breif detail between layer and layer.
```python
nn.shape()
```

#### Forward propagation
The data flow of network is based on **Tensor**. To forward propagation, just past the data to `network.Forward(DATA)` function. And it will return a **list** of **Tensor**.
```python
data = Tensor(1, 3, 28, 28, 0)
result = nn.Forward(data)
```

### Tensor
Tensor in python version is also not completed yet. Just work with basic operation.
#### Initialize the tensor
* Method 1 <br>
You will get a **Tensor t** with identical value **PARAMETER** inside.
```python
t =  Tensor(BATCH, CHANNEL, HEIGHT, WIDTH, PARAMETER);
```

#### Load numpy into tensor
WIth any shape of tensor, it will reshape automatically as `numpy.ndarray`
```python
t = Tensor()
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
t.load_array(arr)    # Tensor shape (1, 2, 2, 2) with value [1, 2, 3 ,4, 5, 6, 7, 8]
print(t)    # Use print to print out the Tensor
```

#### Convert Tensor to numpy
If want to do some data analysis,  you can convert the Tensor to numpy
```python
t = Tensor(1, 2, 2, 2, 1)
arr = t.to_numpy()    # [[[1, 1], [1, 1]], [[1, 1], [1, 1]]]
```

#### Max value and its index
Get the max value and its index inside Tensor
```python
t = Tensor()
arr = np.array([1, 3, 2])
t.load_array(arr)    # Tensor shape (1, 1, 1, 3) with value [1, 3, 2]
index, value = t.max_index()    # value = 3, index = 1
```


## Build and run
#### Linux, MacOS
Just do `make` in the directory. Before make, you can set such options in the `Makefile`:
* `EXEC=EXECUTION_FILE_NAME` You can change the execution file name by yourself.
* `OPENMP=1` to build with OpenMP support to accelerate Network by using multi-core CPU
* `LIBSO=1` to build a library `otter.so`

If your project need to train the YOLOv4 layer, you should revise `OPTS = -Ofast` as `OPTS = -O2` in `Makefile`

#### Windows
If you need to train YOLOv4 layer, you can build with
* `$ g++ -Ofast -fopenmp -o nn *.cpp`

Else, the isnan() function is not working with -Ofast flag
* `$ g++ -O2 -fopenmp -o nn *.cpp`

#### Run
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

    Tensor a(1, 1, 1, 2, 0);
    Tensor b(1, 1, 1, 2, 1);
    Tensor c(1, 1, 1, 2); c = {0, 1};
    Tensor d(1, 1, 1, 2); d = {1, 0};
    vtensor data{a, b, c, d};
    Tensor a_l(1, 1, 1, 1, 0);
    Tensor b_l(1, 1, 1, 1, 0);
    Tensor c_l(1, 1, 1, 1, 1);
    Tensor d_l(1, 1, 1, 1, 1);
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
[5]: https://chrischoy.github.io/research/making-caffe-layer/
[6]: https://arxiv.org/pdf/1604.02878.pdf
[7]: https://arxiv.org/pdf/1804.02767.pdf
[8]: https://www.youtube.com/watch?v=XJ7HLz9VYz0&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh
[9]: https://github.com/pytorch/pytorch



