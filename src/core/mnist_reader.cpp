#include "mnist_reader.h"
#include <string>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std; //Necessario per i cout a l.41,50,59,60
using namespace cv;

// Source - https://stackoverflow.com/a/52406407
// Posted by Jayhello, modified by community. See post 'Timeline' for change history
// Retrieved 2026-10-02, License - CC BY-SA 4.0

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

bool read_mnist_cv(const char* image_filename,
                   const char* label_filename,
                   float*& images,
                   int*& labels,
                   uint32_t& num_items,
                   uint32_t& img_size)
    {
    // Open files
    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);

    if (!image_file.is_open()) {
        std::cerr << "ERRORE: Impossibile aprire il file immagini: " << image_filename << std::endl;
        return false;
    }
    if (!label_file.is_open()) {
        std::cerr << "ERRORE: Impossibile aprire il file labels: " << label_filename << std::endl;
        return false;
    }
    // Read the magic and the meta data
    uint32_t magic = 0;
    uint32_t num_labels = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if(magic != 2051){
        cout<<"Incorrect image file magic: "<<magic<<endl;
        return false;
    }

    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if(magic != 2049){
        cout<<"Incorrect image file magic: "<<magic<<endl;
        return false;
    }

    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    if(num_items != num_labels){
        cout<<"image file nums should equal to label num"<<endl;
        return false;
    }

    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    img_size = rows * cols;

    cout<<"image and label num is: "<<num_items<<endl;
    cout<<"image rows: "<<rows<<", cols: "<<cols<<endl;
    
    images = new float[num_items * img_size];
    labels = new int[num_items];

    unsigned char* temp_pixels = new unsigned char[img_size];
    unsigned char temp_label = 0;

    for (int item_id = 0; item_id < num_items; ++item_id) {
        // read image pixel
        image_file.read(reinterpret_cast<char*>(temp_pixels),img_size);
        // read label
        label_file.read(reinterpret_cast<char*>(&temp_label),1);

        labels[item_id] = static_cast<int>(temp_label);
        uint32_t offset = item_id * img_size;

        for (uint32_t p=0;p<img_size;++p){
            images[offset+p] = static_cast<float>(temp_pixels[p]) / 255.0f;
        }
        //cout<<"lable is: "<<sLabel<<endl;
        //// convert it to cv Mat, and show it
        //cv::Mat image_tmp(rows,cols,CV_8UC1,pixels);
        //// resize bigger for showing
        //cv::resize(image_tmp, image_tmp, cv::Size(100, 100));
        //cv::imshow(sLabel, image_tmp);
        //cv::waitKey(0);
    }

    delete[] temp_pixels;
    return true;
}


