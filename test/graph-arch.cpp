// mnist_train.cpp
#include "../autograd.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <signal.h>

ComputationGraph* pgraph = nullptr; // Global pointer to the computation graph
void signal_handler(int signal){
    if(signal == SIGINT){
        std::cout << "\nTraining interrupted by user." << std::endl;
        pgraph->SaveParams("test/mnist_model_params_interrupt.txt");
        exit(0);
    }
}

int main(int argc, char* argv[]) {
    std::vector<bool> results;
    std::cout << "=== MNIST CNN Training in C++ ===" << std::endl; 
    // 网络参数
    const int n = 28;
    const int n_input = n * n;
    const int n_output = 10;
    const int n_kernel = 32;
    const int n_kernel_2 = 48;

    // 创建网络变量
    auto x = make_input(std::vector<double>(n_input, 0.0), {n, n});
    auto label = make_input(std::vector<double>(n_output, 0.0), {n_output});
    
    // 初始化卷积核权重
    std::random_device rd;
    std::mt19937 gen(42);  // 固定随机种子以确保结果可重现
    std::normal_distribution<double> normal_dist(0.0, 0.1);
    
    auto init_weights = [&](size_t size) {
        std::vector<double> weights(size);
        for (size_t i = 0; i < size; i++) {
            weights[i] = normal_dist(gen);
        }
        return weights;
    };
    MAKE_PARAM(kernel_1, init_weights(3 * 3 * n_kernel), {3, 3, n_kernel});
    MAKE_PARAM(kernel_2, init_weights(3 * 3 * n_kernel_2 * n_kernel), {3, 3, n_kernel, n_kernel_2});
    // auto kernel_1 = make_param(init_weights(3 * 3 * n_kernel), {3, 3, n_kernel});
    // auto kernel_2 = make_param(init_weights(3 * 3 * n_kernel_2 * n_kernel), {3, 3, n_kernel, n_kernel_2});
    
    // 第一个卷积层
    std::cout << "Building first conv layer..." << std::endl;
    std::vector<VarPtr> output_1;
    for (int i = 0; i < n_kernel; i++) {
        // SLICE(kernel_slice, kernel_1, {-1, -1, i});
        // CONV2D(conv_out, x, kernel_slice);
        // RELU(relu_out, conv_out);
        // MAXPOOLING(pool_out, relu_out, 2);
        // output_1.push_back(pool_out);
        auto kernel_slice = slice(kernel_1, {-1, -1, i}, "kernel_slice" + std::to_string(i));  // 提取第i个卷积核
        auto conv_out = conv2d(x, kernel_slice, "conv_out" + std::to_string(i));  // 卷积操作
        auto relu_out = relu(conv_out, "relu_out" + std::to_string(i));  // ReLU激活
        auto pool_out = MaxPooling(relu_out, 2, "pool_out" + std::to_string(i));  // 2x2最大池化
        output_1.push_back(pool_out );
    }
    STACK(feature_maps_1, output_1);
    // auto feature_maps_1 = stack(output_1);  // [32, 13, 13]
    std::cout<<"Feature maps 1 shape: " << feature_maps_1->shape()[0] << "x" << feature_maps_1->shape()[1] << "x" << feature_maps_1->shape()[2] << std::endl;

    // 第二个卷积层
    std::cout << "Building second conv layer..." << std::endl;
    std::vector<VarPtr> slices_1;
    for (int i = 0; i < n_kernel; i++) {
        slices_1.push_back(slice(feature_maps_1, {i, -1, -1}));
    }
    
    std::vector<VarPtr> output_2;
    for (int i = 0; i < n_kernel_2; i++) {
        std::vector<VarPtr> conv_results;
        for (int j = 0; j < n_kernel; j++) {
            SLICE(kernel_slice, kernel_2, {-1, -1, j, i});
            CONV2D(conv_out, slices_1[j], kernel_slice);
            RELU(relu_out, conv_out);
            MAXPOOLING(pool_out, relu_out, 2);
            conv_results.push_back(pool_out);
            // auto kernel_slice = slice(kernel_2, {-1, -1, j, i});
            // auto conv_out = conv2d(slices_1[j], kernel_slice);
            // auto relu_out = relu(conv_out);
            // auto pool_out = MaxPooling(relu_out, 2);
            // conv_results.push_back(pool_out);
        }

        auto summed = sum(conv_results, "summed"+std::to_string(i));
        output_2.push_back(summed);
    }
    auto feature_maps_2 = stack(output_2);
    std::cout<<"Feature maps 2 shape: " << feature_maps_2->shape()[0] << "x" << feature_maps_2->shape()[1] << "x" << feature_maps_2->shape()[2] << std::endl;
    // 全连接层
    std::cout << "Building fully connected layers..." << std::endl;
    size_t input_size = feature_maps_2->size();
    const int mid_size = 128;
    std::cout<<"input size: "<< input_size <<std::endl;
    MAKE_PARAM(W1, init_weights(input_size * mid_size), {input_size, mid_size});
    MAKE_PARAM(b1, init_weights(mid_size), {mid_size});
    MAKE_PARAM(W2, init_weights(mid_size * n_output), {mid_size, n_output});
    MAKE_PARAM(b2, init_weights(n_output), {n_output});
    // auto W1 = make_param(init_weights(input_size * mid_size), {input_size, mid_size});
    // auto b1 = make_param(init_weights(mid_size), {mid_size});
    // auto W2 = make_param(init_weights(mid_size * n_output), {mid_size, n_output});
    // auto b2 = make_param(init_weights(n_output), {n_output});
    std::cout<<"W1 shape: "<< W1->shape()[0] << "x" << W1->shape()[1] <<std::endl;
    std::cout<<"W2 shape: "<< W2->shape()[0] << "x" << W2->shape()[1] <<std::endl;
    FLATTEN(flattened, feature_maps_2);
    MUL(mul1, W1, flattened,0,0);
    ADD(layer1_pre, mul1, b1);
    RELU(layer1, layer1_pre);
    MUL(mul2, W2, layer1, 0, 0);
    ADD(layer2, mul2, b2);
    MSE_LOSS(loss, layer2, label);
    // auto flattened = flatten(feature_maps_2);
    // auto layer1 = relu(add(mul(W1, flattened, 0, 0), b1));
    // auto layer2 = add(mul(W2, layer1, 0, 0), b2);
    // auto loss = mse_loss(layer2, label);
    auto graph = ComputationGraph::BuildFromOutput(loss);
    pgraph = &graph; // 设置全局指针
    pgraph->SaveArch("out/mnist_model_arch.txt");
}