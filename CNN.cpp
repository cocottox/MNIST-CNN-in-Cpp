#include <iostream>
#include <string>
#include <math.h>
#include <vector>

//File principale in cui trascrivere le funzioni del CNN, poi lo renderemo un file.h
//Se vogliamo renderlo cool possiamo lavorare il progetto con assert cosí mano a mano verifichiamo
//il funzionamento di ogni parte, ma non so come si fa il test su c++, cercherò
//ps: nei file header sembra non convenga usare namespace std, quindi aggiungeremo sempre std::
//
//tvb programma nel chill, giusto lascia commenti ogni tanto così sappiamo entrambi cosa e come,
//pure nei commit,
//  Matteo

void softmax(float* arr, size_t dim){
    //Funzione per ottenere una probabilità da ogni elemento di un array basandosi su tutti
    //i valori, noi la usiamo per il vettore di uscita finale ez
    if (dim==0) return;

    float max_val = arr[0];
    for(size_t i=0;i<dim;i++){
        if(arr[i]>max_val){
            max_val = arr[i];
        }
    }
    float sum_exp = 0.0f;
    for(size_t i=0;i<dim;i++){
        arr[i] = std::exp(arr[i]-max_val);
        sum_exp += arr[i];
    }
    for(size_t i=0;i<dim;i++){
        arr[i] /= sum_exp;
    }
}

float cross_entropy(const float* arr, int label){
    //Questa serve a calcolare di quanto si è sbagliato il CNN
    float idx = arr[label];
    const float eps = 1e-7f;
    if (idx < eps){
        idx = eps;
    }
    return -std::log(idx);
}

class ReLu{
    private:
        std::vector<float> X;
    public:
        ReLu(){
        }
        std::vector<float> forward(const std::vector<float> &input_X){
            this->X = input_X;
            std::vector<float> output(input_X.size());
            for (size_t i=0;i<input_X.size();i++){
                output[i] = std::max(0.0f,input_X[i]);
            }
            return output;
        } 
        std::vector<float> backward(const std::vector<float> &dY){
            std::vector<float> dX(dY.size());
            for (size_t i=0;i<dY.size();i++){
                dX[i] = (this->X[i] > 0.0f) ? dY[i] : 0.0f;
            }
            return dX;
        }
};
