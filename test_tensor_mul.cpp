#include <iostream>
#include <vector>
#include "autograd.hpp"
std::vector<double> random_vector(size_t size, double min_val=0.0, double max_val=1.0) {
    std::vector<double> vec(size);
    for (size_t i = 0; i < size; ++i) {
        double random_val = min_val + static_cast<double>(rand()) / RAND_MAX * (max_val - min_val);
        vec[i] = random_val;
    }
    return vec;
}
int main() {
    std::cout << "=== 测试张量乘法功能 ===" << std::endl;
    
    // 测试1: 元素级乘法（广播）
    std::cout << "\n--- 测试1: 元素级乘法 ---" << std::endl;
    auto a1 = make_var({1.0, 2.0, 3.0, 4.0}, true, {2, 2});
    auto b1 = make_var({2.0, 3.0}, true, {2});  // 广播
    
    std::cout << "a1 shape: (2, 2), data: ";
    a1->print();
    std::cout << "b1 shape: (2,), data: ";
    b1->print();
    
    auto result1 = mul(a1, b1);  // 默认元素级乘法
    result1->backward();
    std::cout << "Element-wise multiplication result: ";
    result1->print();
    
    // 测试2: 矩阵乘法（沿最后一维收缩）
    std::cout << "\n--- 测试2: 矩阵乘法 ---" << std::endl;
    auto a2 = make_var({1.0, 2.0, 3.0, 4.0}, true, {2, 2});
    auto b2 = make_var({1.0, 0.0, 0.0, 1.0}, true, {2, 2});
    
    std::cout << "a2 (2x2): ";
    a2->print();
    std::cout << "b2 (2x2): ";
    b2->print();
    
    auto result2 = mul(a2, b2, 1, 0);  // 沿 a 的轴1 和 b 的轴0 收缩
    std::cout << "Matrix multiplication result: ";
    result2->backward();
    result2->print();
    
    // 测试3: 向量内积
    std::cout << "\n--- 测试3: 向量内积 ---" << std::endl;
    auto a3 = make_var({1.0, 2.0, 3.0}, true, {3});
    auto b3 = make_var({4.0, 5.0, 6.0}, true, {3});
    
    std::cout << "a3: ";
    a3->print();
    std::cout << "b3: ";
    b3->print();
    
    auto result3 = mul(a3, b3, 0, 0);  // 沿两个向量的轴0收缩
    result3->backward();
    std::cout << "Vector dot product result: ";
    result3->print();
    
    // 测试4: 反向传播
    std::cout << "\n--- 测试4: 反向传播 ---" << std::endl;
    
    std::vector<double> data = random_vector(20,0.0,10.0);
    
    auto result4 = mul(a4, b4);  // 元素级乘法
    auto loss = sum(result4);    // 求和作为损失
    
    std::cout << "a4: ";
    a4->print();
    std::cout << "b4: ";
    b4->print();
    std::cout << "result4 = a4 * b4: ";
    result4->print();
    
    for(int epoch = 0; epoch < 100; epoch ++ ){
        double learning_rate = 0.01;
        loss->backward();
        a4->update(learning_rate);
        b4->update(learning_rate);
        std::cout<<"Epoch "<< epoch << "loss: "<< loss->item() << std::endl;
    }
    std::cout << "\n=== 测试完成 ===" << std::endl;
    return 0;
}