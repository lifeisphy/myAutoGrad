/**
 * C++ 自动微分框架主程序
 * 基础的自动微分实现，支持标量和向量运算
 */
#include <cstdlib> // for rand()
#include <ctime>   // for time()
#include <algorithm> // for std::transform
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
    std::cout << "C++ 自动微分框架演示" << std::endl;
    std::cout << "====================" << std::endl;
    
    // 基础标量运算示例
    std::cout << "\n1. 基础标量运算:" << std::endl;
    auto x = make_var(2.0, true);
    auto y = make_var(3.0, true);
    
    // 计算 z = x^2 + 2*x*y + y^2
    auto x_squared = mul(x, x);
    auto y_squared = mul(y, y);
    auto xy = mul(x, y);
    auto two_xy = mul(make_var(2.0), xy);
    auto temp = add(x_squared, two_xy);
    auto z = add(temp, y_squared);
    
    std::cout << "z = x^2 + 2*x*y + y^2 = " << z->item() << std::endl;
    
    // 反向传播计算梯度
    z->backward();
    
    std::cout << "∂z/∂x = 2*x + 2*y = " << x->grad_item() << " (期望: " << 2*2 + 2*3 << ")" << std::endl;
    std::cout << "∂z/∂y = 2*x + 2*y = " << y->grad_item() << " (期望: " << 2*2 + 2*3 << ")" << std::endl;
    
    // 激活函数示例
    std::cout << "\n2. 激活函数示例:" << std::endl;
    auto input = make_var(-0.5, true);
    
    auto relu_out = relu(input);
    auto sigmoid_out = sigmoid(input);
    auto tanh_out = tanh_activation(input);
    
    std::cout << "输入: " << input->item() << std::endl;
    std::cout << "ReLU输出: " << relu_out->item() << std::endl;
    std::cout << "Sigmoid输出: " << sigmoid_out->item() << std::endl;
    std::cout << "Tanh输出: " << tanh_out->item() << std::endl;
    
    // 向量运算示例
    std::cout << "\n3. 向量运算示例:" << std::endl;
    std::vector<double> vec1_data = {1.0, 2.0, 3.0};
    std::vector<double> vec2_data = {0.5, 1.5, 2.5};
    
    auto vec1 = make_var(vec1_data, true);
    auto vec2 = make_var(vec2_data, true);
    
    auto vec_sum = add(vec1, vec2);
    auto vec_product = mul(vec1, vec2);
    
    std::cout << "向量1: [1.0, 2.0, 3.0]" << std::endl;
    std::cout << "向量2: [0.5, 1.5, 2.5]" << std::endl;
    std::cout << "向量加法: [";
    for (size_t i = 0; i < vec_sum->data().size(); ++i) {
        std::cout << vec_sum->data()[i];
        if (i < vec_sum->data().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "向量乘法: [";
    for (size_t i = 0; i < vec_product->data().size(); ++i) {
        std::cout << vec_product->data()[i];
        if (i < vec_product->data().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // 损失函数示例
    std::cout << "\n4. 损失函数示例:" << std::endl;
    std::vector<double> predictions_data = {0.8, 0.3, 0.9};
    std::vector<double> targets_data = {1.0, 0.0, 1.0};
    
    auto predictions = make_var(predictions_data, true);
    auto targets = make_var(targets_data, false);
    
    auto mse = mse_loss(predictions, targets);
    auto bce = binary_cross_entropy_loss(predictions, targets);
    
    std::cout << "预测值: [0.8, 0.3, 0.9]" << std::endl;
    std::cout << "目标值: [1.0, 0.0, 1.0]" << std::endl;
    std::cout << "MSE损失: " << mse->item() << std::endl;
    std::cout << "BCE损失: " << bce->item() << std::endl;
    
    // 简单神经网络示例
    std::cout << "\n5. 简单神经网络训练示例:" << std::endl;
    
    // 网络参数
    auto w = make_var(0.1, true);
    auto b = make_var(0.0, true);
    
    // 训练数据
    // random generate data
    // include <cstdlib>
    std::vector<double> x_train = random_vector(20, 0.0, 10.0);
    std::vector<double> y_train(x_train.size());
    std::transform(x_train.begin(), x_train.end(), y_train.begin(), [](double x) { return 2.0 * x; });
    // std::vector<double> x_train = {1.0, 2.0, 3.0, 4.0};
    // std::vector<double> y_train = {2.0, 4.0, 6.0, 8.0};  // y = 2x
    
    double learning_rate = 0.01;
    
    std::cout << "训练目标: 学习函数 y = 2x" << std::endl;
    std::cout << "初始参数: w=" << w->item() << ", b=" << b->item() << std::endl;
    
    // 训练循环
    for (int epoch = 0; epoch < 200; ++epoch) {
        double total_loss = 0.0;
        
        // 每个数据点
        for (size_t i = 0; i < x_train.size(); ++i) {
            // 清零梯度
            w->zero_grad();
            b->zero_grad();
            
            // 前向传播
            auto x_input = make_var(x_train[i], false);
            auto y_target = make_var(y_train[i], false);
            
            auto y_pred = add(mul(w, x_input), b);
            auto loss = mse_loss(y_pred, y_target);
            
            total_loss += loss->item();
            
            // 反向传播
            loss->backward();
            
            // 更新参数
            // auto new_w_data = w->data()[0] - learning_rate * w->grad()[0];
            // auto new_b_data = b->data()[0] - learning_rate * b->grad()[0];
            // only update data without creating new Variable
            // w = make_var(new_w_data, true);
            w->update(learning_rate);
            b->update(learning_rate);
            // w->data()[0] = new_w_data;
            // // b = make_var(new_b_data, true);
            // b->data()[0] = new_b_data;
        }
        
        if (epoch % 20 == 0) {
            std::cout << "Epoch " << epoch << ", 平均损失: " << total_loss / x_train.size() 
                      << ", w=" << w->item() << ", b=" << b->item() << std::endl;
        }
    }
    
    std::cout << "\n训练完成!" << std::endl;
    std::cout << "最终参数: w=" << w->item() << ", b=" << b->item() << std::endl;
    std::cout << "期望参数: w=2.0, b=0.0" << std::endl;
    
    // 测试预测
    std::cout << "\n6. 预测测试:" << std::endl;
    auto test_x = make_var(5.0, false);
    auto prediction = add(mul(w, test_x), b);
    std::cout << "输入 x=5.0, 预测 y=" << prediction->item() << " (期望: 10.0)" << std::endl;
    
    return 0;
}