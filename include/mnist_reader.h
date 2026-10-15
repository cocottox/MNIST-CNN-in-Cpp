#ifndef MNIST_READ_H
#define MNIST_READ_H
#include <cstdint>

bool read_mnist_cv(const char* image_filename, 
                   const char* label_filename,
                   float*& images,
                   int*& labels,
                   uint32_t& num_items,
                   uint32_t& img_size);

#endif
