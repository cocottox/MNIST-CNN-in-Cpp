#include <iostream>
#include <string>
#include "mnist_reader.h"


int main(){
    float* train_images = nullptr;
    int* train_labels = nullptr;
    uint32_t num_items = 0;
    uint32_t img_size = 0;
    
    std::cout << "Programma partito" << std::endl;

    std::string base_dir = "C:/Users/MaoLaptop/Desktop/Cpp/CNN/MNIST_ORG/";
    std::string img_path = base_dir + "train-images.idx3-ubyte";
    std::string label_path = base_dir + "train-labels.idx1-ubyte";

    if(!read_mnist_cv(img_path.c_str(),label_path.c_str(),train_images,train_labels,num_items,img_size)){
        return -1;
    }
    int idx=0;
    //Prima immagine è un vettore di tutte le immagini
    //sappiamo che il size di ogni foto è 28x28 quindi ci muoveremo con un indice attraverso questo vettore di 28x28 volte per ogni immagine
    float* prima_immagine = &train_images[idx * img_size];
    int primo_label = train_labels[idx];

    std::cout<< "Primo label --> " << primo_label << std::endl;
    //Questo for va tolto, è solo per mostrare l'immagine a terminale con 1 e 0
    for(int i=0;i<img_size;i++){
        if(prima_immagine[i]>0){prima_immagine[i]=1;}
        std::cout << prima_immagine[i];
        //sto if è solo per andare a capo e renderla leggibile ogni 28esimo pixel della riga
        if(i % 28 == 0 & i!=0){std::cout<<std::endl;}
    }
    
    return 0;
};
