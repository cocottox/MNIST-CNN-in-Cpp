#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "../include/CNN.h"

#include <stdio.h> 
#include <vector>


TEST_CASE("--- Test Funzione softmax ---","[softmax test]"){
    std::vector<float> vect1 = {0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,0.0f};
    softmax(vect1);

    float sum1 = 0.0f;
    for(float val : vect1){
        REQUIRE(val>=0.0f);
        REQUIRE(val<=1.0f);
        sum1 += val;
    }

    REQUIRE(sum1 == Catch::Approx(1.0f).margin(0.0001f));

    std::vector<float> vect2 = {1.0f,1.0f,100.0f};
    softmax(vect2);

    float sum2 = 0.0f;
    for(float val : vect2){
        REQUIRE(val>=Catch::Approx(0.0f).margin(0.00001f));
        REQUIRE(val<=Catch::Approx(1.0f).margin(0.00001f));
        sum2 += val;
    }

    REQUIRE(sum2 == Catch::Approx(1.0f).margin(0.0001f));

    std::vector<float> vect3 = {2.0f,2.0f,2.0f};
    softmax(vect3);

    float sum3 = 0.0f;
    for(float val : vect3){
        REQUIRE(val>=0.0f);
        REQUIRE(val<=1.0f);
        sum3 += val;
    }

    REQUIRE(sum3 == Catch::Approx(1.0f).margin(0.0001f));
    
    std::vector<float> vect4 = {0.1f,0.5f,-2.0f,6.0f};
    softmax(vect4);

    float sum4 = 0.0f;
    for(float val : vect4){
        REQUIRE(val>=0.0f);
        REQUIRE(val<=1.0f);
        sum4 += val;
    }

    REQUIRE(sum4 == Catch::Approx(1.0f).margin(0.0001f));
}

TEST_CASE("--- ReLu Class TEst","[ReLu Test]"){
    // Test for the forward
    ReLu rl;     

    SECTION("Test del forward"){
        std::vector<float> vect5 = {0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,0.0f};
        std::vector<float> ReLu_vect5 = rl.forward(vect5);
        for(size_t i = 0;i<vect5.size();i++){
            if(vect5[i] > 0){
                REQUIRE(vect5[i] == ReLu_vect5[i]);
            } else{
                REQUIRE(ReLu_vect5[i] == 0.0f);
            }
        }

        std::vector<float> vect6 = {0.1f,0.5f,-2.0f,6.0f};   
        std::vector<float> ReLu_vect6 = rl.forward(vect6);
        for(size_t i = 0;i<vect6.size();i++){
            if(vect6[i] > 0){
                REQUIRE(vect6[i] == ReLu_vect6[i]);
            } else{
                REQUIRE(ReLu_vect6[i] == 0.0f);
            }
        }

        std::vector<float> vect7 = {-1.0f,-1.0f,-1.0f,-1.0f,-1.0f};   
        std::vector<float> ReLu_vect7 = rl.forward(vect7);
        for(size_t i = 0;i<vect7.size();i++){
            if(vect7[i] > 0){
                REQUIRE(vect7[i] == ReLu_vect7[i]);
            } else{
                REQUIRE(ReLu_vect7[i] == 0.0f);
            }
        }
    }
    //SECTION("Test del backward"){
    //    
    //}
}
