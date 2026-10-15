#pragma once
#ifndef CNN_H
#define CNN_H

#include <cmath>
#include <vector>


void softmax(float* arr, size_t dim);   

float cross_entropy(const float* arr, int label);

class ReLu{
    private:
        std::vector <float> X;
    public:
        ReLu();
        std::vector<float> forward(const std::vector<float> &input_X);
        std::vector<float> backward(const std::vector<float> &dY);
};

#endif
