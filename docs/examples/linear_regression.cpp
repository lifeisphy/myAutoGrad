#include "../../autograd.hpp"
#include <iostream>
#include <vector>
#include <random>

int main() {
    // 生成 synthetic 数据
    std::vector<double> X = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.1, 4.3, 6.1, 8.0, 10.2};  // y ≈ 2x + 0.1
    
    // 创建变量
    auto x = make_input(0.0);
    auto target = make_input(0.0);
    auto w = make_param(0.1);  // 权重
    auto b = make_param(0.1);  // 偏置
    
    // 构建模型：y_pred = w * x + b
    auto y_pred = add(mul(w, x), b);
    auto loss = mse_loss(y_pred, target);
    
    // 训练
    double learning_rate = 0.01;
    for (int epoch = 0; epoch < 100; ++epoch) {
        double total_loss = 0.0;
        
        for (size_t i = 0; i < X.size(); ++i) {
            x->set_input(X[i]);
            target->set_input(y[i]);
            
            loss->zero_grad_recursive();
            loss->calc();
            loss->backward();
            
            w->update(learning_rate);
            b->update(learning_rate);
            
            total_loss += loss->item();
        }
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / X.size() 
                      << ", w: " << w->item() << ", b: " << b->item() << std::endl;
        }
    }
    
    // 测试
    x->set_input(6.0);
    y_pred->calc();
    std::cout << "Prediction for x=6: " << y_pred->item() << std::endl;
    
    return 0;
}