#include "../../autograd.hpp"
#include <iostream>
#include <vector>
#include <random>

int main() {
    // 网络参数
    const int input_size = 2;
    const int hidden_size = 4;
    const int output_size = 1;
    
    // 创建变量
    auto x = make_input(std::vector<double>(input_size, 0.0), {input_size});
    auto target = make_input(0.0);
    
    // 第一层
    auto W1 = make_param(vec_r(input_size * hidden_size), {hidden_size, input_size});
    auto b1 = make_param(vec_r(hidden_size), {hidden_size});
    
    // 第二层
    auto W2 = make_param(vec_r(hidden_size * output_size), {output_size, hidden_size});
    auto b2 = make_param(vec_r(output_size), {output_size});
    auto z1 = add(mul(W1, x, 1, 0), b1);
    auto a1 = relu(z1);
    auto z2 = add(mul(W2, a1, 1, 0), b2);
    auto output = sigmoid(z2);
    auto loss = binary_cross_entropy_loss(output, target);
    
    // XOR 数据
    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<double> y = {0, 1, 1, 0};
    
    // 训练
    double learning_rate = 0.1;
    for (int epoch = 0; epoch < 5000; ++epoch) {
        double total_loss = 0.0;
        
        for (size_t i = 0; i < X.size(); ++i) {
            x->set_input(X[i]);
            target->set_input(y[i]);
            
            loss->zero_grad_recursive();
            loss->calc();
            loss->backward();
            
            W1->update(learning_rate);
            b1->update(learning_rate);
            W2->update(learning_rate);
            b2->update(learning_rate);
            
            total_loss += loss->item();
        }
        
        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / X.size() << std::endl;
        }
    }
    
    // 测试
    std::cout << "\nTesting XOR:" << std::endl;
    for (size_t i = 0; i < X.size(); ++i) {
        x->set_input(X[i]);
        output->calc();
        std::cout << X[i][0] << " XOR " << X[i][1] << " = "
                  << (output->item() > 0.5 ? 1 : 0)
                  << " (prob: " << output->item() << ")" << std::endl;
    }
    
    return 0;
}