#pragma once
#include <vector>

void softmax(float* arr, size_t dim);   

float cross_entropy(const float* arr, int label){return -std::log(arr[label]);}

class Relu{
    private:
        std::vector <float> X;
    public:
        ReLu();
        std::vector<float> forward(const std::vector<float> &input_X);
        std::vector<float> backward(const std::vector<float> &dY);
};
