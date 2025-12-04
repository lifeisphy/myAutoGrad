#include "../../autograd.hpp"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    // 二分类数据
    std::vector<std::vector<double>> X = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 3.0}, {4.0, 5.0},  // 正类
        {1.0, 1.0}, {2.0, 1.0}, {3.0, 1.0}, {4.0, 2.0}   // 负类
    };
    std::vector<double> y = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    
    // 创建变量
    auto x1 = make_input(0.0);
    auto x2 = make_input(0.0);
    auto target = make_input(0.0);
    auto w1 = make_param(0.1);
    auto w2 = make_param(0.1);
    auto b = make_param(0.1);
    
    // 构建模型：logits = w1*x1 + w2*x2 + b
    auto logits = add(add(mul(w1, x1), mul(w2, x2)), b);
    auto probs = sigmoid(logits);
    auto loss = binary_cross_entropy_loss(probs, target);
    
    // 训练
    double learning_rate = 0.1;
    for (int epoch = 0; epoch < 1000; ++epoch) {
        double total_loss = 0.0;
        
        for (size_t i = 0; i < X.size(); ++i) {
            x1->set_input(X[i][0]);
            x2->set_input(X[i][1]);
            target->set_input(y[i]);
            
            loss->zero_grad_recursive();
            loss->calc();
            loss->backward();
            
            w1->update(learning_rate);
            w2->update(learning_rate);
            b->update(learning_rate);
            
            total_loss += loss->item();
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / X.size() << std::endl;
        }
    }
    
    // 测试
    x1->set_input(2.5);
    x2->set_input(2.5);
    probs->calc();
    std::cout << "Prediction for (2.5, 2.5): " << probs->item() << std::endl;
    
    return 0;
}