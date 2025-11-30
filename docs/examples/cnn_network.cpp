#include "../../autograd.hpp"
#include <iostream>
#include <vector>

void build_cnn() {
    // 输入层
    auto x = make_input(std::vector<double>(28 * 28, 0.0), {28, 28});
    
    // 第一个卷积层
    auto conv1_weights = make_param(vec_r(3 * 3 * 32), {3, 3, 32});
    std::vector<VarPtr> conv1_outputs;
    for (int i = 0; i < 32; i++) {
        auto kernel_slice = slice(conv1_weights, {-1, -1, i});
        auto conv_out = conv2d(x, kernel_slice);
        auto relu_out = relu(conv_out);
        auto pool_out = MaxPooling(relu_out, 2);
        conv1_outputs.push_back(pool_out);
    }
    auto feature_maps1 = stack(conv1_outputs);
    
    // 第二个卷积层
    auto conv2_weights = make_param(vec_r(3 * 3 * 32 * 64), {3, 3, 32, 64});
    std::vector<VarPtr> conv2_outputs;
    for (int i = 0; i < 64; i++) {
        std::vector<VarPtr> conv_results;
        for (int j = 0; j < 32; j++) {
            auto feature_slice = slice(feature_maps1, {j, -1, -1});
            auto kernel_slice = slice(conv2_weights, {-1, -1, j, i});
            auto conv_out = conv2d(feature_slice, kernel_slice);
            auto relu_out = relu(conv_out);
            auto pool_out = MaxPooling(relu_out, 2);
            conv_results.push_back(pool_out);
        }
        auto summed = sum(conv_results);
        conv2_outputs.push_back(summed);
    }
    auto feature_maps2 = stack(conv2_outputs);
    
    // 全连接层
    auto flattened = flatten(feature_maps2);
    auto fc1_weights = make_param(vec_r(flattened->size() * 128), {flattened->size(), 128});
    auto fc1_bias = make_param(vec_r(128), {128});
    auto fc2_weights = make_param(vec_r(128 * 10), {128, 10});
    auto fc2_bias = make_param(vec_r(10), {10});
    
    auto fc1 = add(mul(fc1_weights, flattened, 0, 0), fc1_bias);
    auto fc1_relu = relu(fc1);
    auto fc2 = add(mul(fc2_weights, fc1_relu, 0, 0), fc2_bias);
    
    // 输出层
    auto target = make_input(std::vector<double>(10, 0.0), {10});
    auto loss = mse_loss(fc2, target);
    
    // 构建计算图
    auto graph = ComputationGraph::BuildFromOutput(loss);
    
    // 训练循环...
}

int main() {
    build_cnn();
    std::cout << "CNN structure built successfully!" << std::endl;
    return 0;
}