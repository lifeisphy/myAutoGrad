#include "../../autograd.hpp"
#include <iostream>
#include <vector>

int main() {
    // 构建简单的神经网络
    auto x = make_input(0.0);
    auto target = make_input(0.0);
    auto w = make_param(0.1);
    auto b = make_param(0.1);
    
    auto y_pred = add(mul(w, x), b);
    auto loss = mse_loss(y_pred, target);
    
    // 创建计算图
    auto graph = ComputationGraph::BuildFromOutput(loss);
    
    // 创建 Adam 优化器
    AdamOptimizer optimizer(0.01);
    optimizer.set_parameter_nodes(graph.parameter_nodes);
    
    // 生成数据
    std::vector<double> X = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.1, 4.3, 6.1, 8.0, 10.2};
    
    // 训练
    for (int epoch = 0; epoch < 100; ++epoch) {
        double total_loss = 0.0;
        
        for (size_t i = 0; i < X.size(); ++i) {
            x->set_input(X[i]);
            target->set_input(y[i]);
            
            loss->zero_grad_recursive();
            loss->calc();
            loss->backward();
            
            // 使用 Adam 优化器更新参数
            for (auto& param : graph.parameter_nodes) {
                optimizer.update(param);
            }
            
            total_loss += loss->item();
        }
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / X.size() << std::endl;
        }
    }
    
    return 0;
}