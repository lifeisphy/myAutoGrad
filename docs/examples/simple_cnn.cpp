#include "../../autograd.hpp"
#include <iostream>
#include <vector>

int main() {
    // 创建简单的 5x5 输入图像
    auto input = make_input(std::vector<double>{
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    }, {5, 5});
    
    // 创建 3x3 卷积核
    auto kernel = make_param(std::vector<double>{
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    }, {3, 3});
    
    // 卷积操作
    auto conv_result = conv2d(input, kernel);
    auto pooled_result = MaxPooling(conv_result, 2);
    
    // 前向计算
    pooled_result->calc();
    
    std::cout << "Input shape: ";
    print_vec(std::cout, input->shape());
    std::cout << std::endl;
    
    std::cout << "Conv result shape: ";
    print_vec(std::cout, conv_result->shape());
    std::cout << std::endl;
    
    std::cout << "Pooled result shape: ";
    print_vec(std::cout, pooled_result->shape());
    std::cout << std::endl;
    
    std::cout << "Pooled result: ";
    pooled_result->print(std::cout, true);
    
    return 0;
}