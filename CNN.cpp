#include <iostream>
#include <string>
#include "mnist_reader.h"


int main(){
    std::string base_dir = "C:/Users/MaoLaptop/Desktop/Cpp/CNN/MNIST_ORG/";
    std::string img_path = base_dir + "train-images-idx3-ubyte";
    std::string label_path = base_dir + "train-labels-idx1-ubyte";

    read_mnist_cv(img_path.c_str(),label_path.c_str());
    return 0;
};
