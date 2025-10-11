/**
 * 自动微分框架测试示例
 * 演示基础运算、激活函数、损失函数和梯度计算
 */

#include "../autograd.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
using vec=std::vector<double>;
// 数值梯度检查函数
double numerical_gradient(std::function<double(double)> func, double x, double h = 1e-5) {
    return (func(x + h) - func(x - h)) / (2 * h);
}
void test_operations(){
    auto a = make_var(vec{1.0,2.0,3.0,4.0}, {2,1,2});
    auto b = make_var(vec{1,2},{2,1});
    auto c = add(a,b);
    c->calc();
    c->backward();
    c->print();
    a->print();
    b->print();
}
// 测试基础运算
void test_basic_operations() {
    std::cout << "\n=== 测试基础运算 ===" << std::endl;
    
    // 测试加法
    auto a = make_var(2.0);
    auto b = make_var(3.0);
    auto c = add(a, b);
    c->calc();
    c->backward();
    
    std::cout << "加法测试: 2 + 3 = " << c->item() << std::endl;
    std::cout << "a的梯度: " << a->grad_item() << " (期望: 1.0)" << std::endl;
    std::cout << "b的梯度: " << b->grad_item() << " (期望: 1.0)" << std::endl;
    
    // 测试乘法
    a->zero_grad();
    b->zero_grad();
    auto d = mul(a, b);
    d->calc();
    d->backward();
    
    std::cout << "\n乘法测试: 2 * 3 = " << d->item() << std::endl;
    std::cout << "a的梯度: " << a->grad_item() << " (期望: 3.0)" << std::endl;
    std::cout << "b的梯度: " << b->grad_item() << " (期望: 2.0)" << std::endl;
}

// 测试复合函数
void test_composite_function() {
    std::cout << "\n=== 测试复合函数 ===" << std::endl;
    
    // 测试 f(x) = x^2 + 2*x + 1 在 x=3 处
    auto x = make_var(3.0);
    auto x_squared = mul(x, x);
    auto two_x = mul(make_var(2.0), x);
    auto temp = add(x_squared, two_x);
    auto y = add(temp, make_var(1.0));
    y->calc();
    y->backward();
    
    std::cout << "f(3) = 3^2 + 2*3 + 1 = " << y->item() << " (期望: 16)" << std::endl;
    std::cout << "f'(3) = 2*3 + 2 = " << x->grad_item() << " (期望: 8.0)" << std::endl;
    
    // 数值验证
    auto func = [](double val) {
        return val * val + 2 * val + 1;
    };
    double numerical_grad = numerical_gradient(func, 3.0);
    std::cout << "数值验证梯度: " << numerical_grad << std::endl;
}

// 测试激活函数
void test_activation_functions() {
    std::cout << "\n=== 测试激活函数 ===" << std::endl;
    
    // 测试ReLU
    auto x1 = make_var(2.0);
    auto relu_result = relu(x1);
    relu_result->calc();
    relu_result->backward();
    
    std::cout << "ReLU(2.0) = " << relu_result->item() << " (期望: 2.0)" << std::endl;
    std::cout << "ReLU'(2.0) = " << x1->grad_item() << " (期望: 1.0)" << std::endl;
    
    // 测试负值的ReLU
    auto x2 = make_var(-1.0);
    auto relu_result2 = relu(x2);
    relu_result2->calc();
    relu_result2->backward();
    
    std::cout << "ReLU(-1.0) = " << relu_result2->item() << " (期望: 0.0)" << std::endl;
    std::cout << "ReLU'(-1.0) = " << x2->grad_item() << " (期望: 0.0)" << std::endl;
    
    // 测试Sigmoid
    auto x3 = make_var(0.0);
    auto sigmoid_result = sigmoid(x3);
    sigmoid_result->calc();
    sigmoid_result->backward();
    
    std::cout << "\\nSigmoid(0.0) = " << sigmoid_result->item() << " (期望: 0.5)" << std::endl;
    std::cout << "Sigmoid'(0.0) = " << x3->grad_item() << " (期望: 0.25)" << std::endl;
    
    // 测试Tanh
    auto x4 = make_var(0.0);
    auto tanh_result = tanh_activation(x4);
    tanh_result->calc();
    tanh_result->backward();
    
    std::cout << "Tanh(0.0) = " << tanh_result->item() << " (期望: 0.0)" << std::endl;
    std::cout << "Tanh'(0.0) = " << x4->grad_item() << " (期望: 1.0)" << std::endl;
}

// 测试向量运算
void test_vector_operations() {
    std::cout << "\n=== 测试向量运算 ===" << std::endl;
    
    // 创建向量
    std::vector<double> data1 = {1.0, 2.0, 3.0};
    std::vector<double> data2 = {4.0, 5.0, 6.0};
    
    auto vec1 = make_var(data1);
    auto vec2 = make_var(data2);
    
    // 向量加法
    auto vec_sum = add(vec1, vec2);
    auto sum_result = sum(vec_sum);  // 求总和
    sum_result->calc();
    sum_result->backward();
    
    std::cout << "向量求和结果: " << sum_result->item() << " (期望: 21.0)" << std::endl;
    
    std::cout << "vec1梯度: [";
    for (size_t i = 0; i < vec1->grad().size(); ++i) {
        std::cout << vec1->grad()[i];
        if (i < vec1->grad().size() - 1) std::cout << ", ";
    }
    std::cout << "] (期望: [1, 1, 1])" << std::endl;
    
    // 测试广播
    vec1->zero_grad();
    vec2->zero_grad();
    
    auto scalar = make_var(2.0);
    auto broadcast_mul = mul(vec1, scalar);
    auto broadcast_sum = sum(broadcast_mul);
    broadcast_sum->calc();
    broadcast_sum->backward();
    
    std::cout << "\\n广播乘法求和: " << broadcast_sum->item() << " (期望: 12.0)" << std::endl;
    std::cout << "标量梯度: " << scalar->grad_item() << " (期望: 6.0)" << std::endl;
}

// 测试损失函数
void test_loss_functions() {
    std::cout << "\n=== 测试损失函数 ===" << std::endl;
    
    // 测试MSE损失
    std::vector<double> pred_data = {1.0, 2.0, 3.0};
    std::vector<double> target_data = {1.5, 2.5, 2.5};
    
    auto predictions = make_var(pred_data);
    auto targets = make_var(target_data);
    
    auto mse = mse_loss(predictions, targets);
    mse->calc();
    mse->backward();
    
    std::cout << "MSE损失: " << mse->item() << std::endl;
    
    std::cout << "预测值梯度: [";
    for (size_t i = 0; i < predictions->grad().size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << predictions->grad()[i];
        if (i < predictions->grad().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // 测试二元交叉熵损失
    std::vector<double> prob_data = {0.7, 0.3, 0.8};
    std::vector<double> label_data = {1.0, 0.0, 1.0};
    
    auto probs = make_var(prob_data);
    auto labels = make_var(label_data);
    
    auto bce = binary_cross_entropy_loss(probs, labels);
    bce->calc();
    bce->backward();
    
    std::cout << "\\n二元交叉熵损失: " << bce->item() << std::endl;
    
    std::cout << "概率梯度: [";
    for (size_t i = 0; i < probs->grad().size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << probs->grad()[i];
        if (i < probs->grad().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// 简单神经网络示例
void test_simple_neural_network() {
    std::cout << "\n=== 简单神经网络示例 ===" << std::endl;
    
    // 输入数据
    auto x = make_var(0.5);
    
    // 权重和偏置
    auto w1 = make_var(0.3);
    auto b1 = make_var(0.1);
    auto w2 = make_var(0.7);
    auto b2 = make_var(0.2);
    
    // 前向传播: x -> linear -> sigmoid -> linear -> sigmoid
    auto z1 = add(mul(x, w1), b1);
    auto a1 = sigmoid(z1);
    auto z2 = add(mul(a1, w2), b2);
    auto output = sigmoid(z2);
    
    // 目标值
    auto target = make_var(0.8);
    
    // 计算损失
    auto loss = mse_loss(output, target);
    
    std::cout << "网络输出: " << output->item() << std::endl;
    std::cout << "损失: " << loss->item() << std::endl;
    
    // 反向传播
    loss->backward();
    
    std::cout << "\\n梯度:" << std::endl;
    std::cout << "w1梯度: " << w1->grad_item() << std::endl;
    std::cout << "b1梯度: " << b1->grad_item() << std::endl;
    std::cout << "w2梯度: " << w2->grad_item() << std::endl;
    std::cout << "b2梯度: " << b2->grad_item() << std::endl;
}

int main() {
    std::cout << "C++ 自动微分框架测试" << std::endl;
    std::cout << "========================" << std::endl;
    
    try {
        test_operations();
        // test_basic_operations();
        // test_composite_function();
        // test_activation_functions();
        // test_vector_operations();
        // test_loss_functions();
        // test_simple_neural_network();
        
        std::cout << "\n✅ 所有测试完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}