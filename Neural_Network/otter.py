from ctypes import *
import numpy as np

lib = CDLL("otter.so", RTLD_GLOBAL)
lib.main.argtypes = c_int, POINTER(c_char_p)
main = lib.main

Neural_Network_Handle = POINTER(c_char)
Tensor_Handle = POINTER(c_char)

lib.create_network.argtypes = [c_char_p]
lib.create_network.restype = Neural_Network_Handle

lib.free_network.argtypes = [Neural_Network_Handle]

lib.network_load_ottermodel.argtypes = [Neural_Network_Handle, c_char_p]

lib.network_forward.argtypes = [Neural_Network_Handle, Tensor_Handle]
lib.network_forward.restype = POINTER(Tensor_Handle)

lib.network_shape.argtypes = [Neural_Network_Handle]

lib.network_getoutputnum.argtypes = [Neural_Network_Handle]
lib.network_getoutputnum.restype = c_int

lib.create_tensor_init.argtypes = [c_int, c_int, c_int, c_int, c_float]
lib.create_tensor_init.restype = Tensor_Handle

lib.create_tensor_array.argtypes = [POINTER(c_float), c_int, c_int, c_int]
lib.create_tensor_array.restype = Tensor_Handle

lib.copy_tensor.argtypes = [Tensor_Handle]
lib.copy_tensor.restype = Tensor_Handle

lib.free_tensor.argtypes = [Tensor_Handle]

lib.tensor_batch.argtypes = [Tensor_Handle]
lib.tensor_batch.restype = c_int

lib.tensor_channel.argtypes = [Tensor_Handle]
lib.tensor_channel.restype = c_int

lib.tensor_height.argtypes = [Tensor_Handle]
lib.tensor_height.restype = c_int

lib.tensor_width.argtypes = [Tensor_Handle]
lib.tensor_width.restype = c_int

lib.tensor_get.argtypes = [Tensor_Handle, c_int]
lib.tensor_get.restype = c_float

lib.tensor_set.argtypes = [Tensor_Handle, c_int, c_float]

lib.tensor_get_weight.argtypes = [Tensor_Handle]
lib.tensor_get_weight.restype = POINTER(c_float)

class Neural_Network:
    def __init__(self, network_name = ''):
        self.network_name = network_name
        self.output_num = 0
        self.instance = lib.create_network(network_name.encode("ascii"))
        
    def __del__(self):
        lib.free_network(self.instance)
        
    def load_ottermodel(self, model):
        lib.network_load_ottermodel(self.instance, model.encode("ascii"))
        self.output_num = lib.network_getoutputnum(self.instance)
        
    def shape(self):
        lib.network_shape(self.instance)
        
    def Forward(self, tensor):
        result = lib.network_forward(self.instance, tensor.instance)
        result_tensor = []
        for i in range(self.output_num):
            item = Tensor()
            item.load_instance(result[i])
            result_tensor.append(item)
        return result_tensor
        
class Tensor:
    def __init__(self, batch = 0, channel = 0, height = 0, width = 0, parameter = 0):
        self.batch = batch
        self.channel = channel
        self.height = height
        self.width = width
        self.instance = lib.create_tensor_init(batch, channel, height, width, parameter)
        
    def __del__(self):
        lib.free_tensor(self.instance)
        
    def __str__(self):
        lib.tensor_show(self.instance)
        return "[%d, %d, %d, %d]" % (self.batch, self.channel, self.height, self.width)

    def __getitem__(self, key):
        return lib.tensor_get(self.instance, key)

    def __setitem__(self, key, value):
        lib.tensor_set(self.instance, key, value)

    def max_index(self):
        data = self.to_numpy()
        return np.argmax(data), np.amax(data)

    def to_numpy(self):
        return np.ctypeslib.as_array(lib.tensor_get_weight(self.instance), shape=(self.batch, self.channel, self.height, self.width))

    def shape(self):
        print("[%d, %d, %d, %d]" % (self.batch, self.channel, self.height, self.width))
        
    def load_instance(self, object):
        lib.free_tensor(self.instance)
        self.batch = lib.tensor_batch(object)
        self.channel = lib.tensor_channel(object)
        self.height = lib.tensor_height(object)
        self.width = lib.tensor_width(object)
        self.instance = lib.copy_tensor(object)
        
    def load_array(self, array):
        self.batch = 1
        self.channel = 1
        self.height = 1
        self.width = 1
        
        if (array.ndim == 1):
            self.width = array.shape[0]
        elif (array.ndim == 2):
            self.width = array.shape[1]
            self.height = array.shape[0]
        elif (array.ndim == 3):
            self.width = array.shape[2]
            self.height = array.shape[1]
            self.channel = array.shape[0]
        
        lib.free_tensor(self.instance)
        if isinstance(array, np.float32) is False:
            array = array.astype('float32')

        self.instance = lib.create_tensor_array(array.ctypes.data_as(POINTER(c_float)), self.width, self.height, self.channel)
        
        
import cv2
import matplotlib.pyplot as plt

label_dict={0:"bee",1:"cat",2:"fish"}

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
   fig=plt.gcf()
   fig.set_size_inches(12, 14)
   if num > 25: num=25
   for i in range(0, num):
       ax=plt.subplot(5, 5, i+1)
       ax.imshow(images[i], cmap='binary')
       title=str(idx[i]) + "." + label_dict[idx[i]] + " (" + "{0:0.2f}".format(prediction[i]) + ")"
       ax.set_title(title, fontsize=10)
       ax.set_xticks([]);
       ax.set_yticks([]);
   plt.show()

if __name__ == "__main__":
    #args = (c_char_p * 2)(b'backup_50048_v4_r2.ottermodel', b'sheep.jpg')
    #main(len(args), args)
    nn = Neural_Network()
    nn.load_ottermodel('test_all_layer.ottermodel')
    nn.shape()

    images = []
    labels = []
    prediction = []
    idx = []

    cat = np.load('cat.npy')
    fish = np.load('fish.npy')
    bee = np.load('bee.npy')

    data = np.concatenate((cat, fish, bee))
    print(data.shape)
    np.random.shuffle(data)

    for i in range(25):
        a = data[i].reshape((28, 28))

        c = np.array([a, a, a])
        test = Tensor()
        test.load_array(c)

        result = nn.Forward(test)
        index, prob = result[0].max_index()
        images.append(a)
        idx.append(index)
        prediction.append(prob)
        # print(result[0].max_index())


    plot_images_labels_prediction(images, labels, prediction, idx, 25)
