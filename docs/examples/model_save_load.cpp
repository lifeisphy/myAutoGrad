#include "../../autograd.hpp"
#include <iostream>

int main() {
    // 构建简单模型
    auto x = make_input(0.0);
    auto w = make_param(0.5);
    auto b = make_param(0.1);
    auto y = add(mul(w, x), b);
    auto loss = mse_loss(y, make_var(1.0));
    
    auto graph = ComputationGraph::BuildFromOutput(loss);
    
    // 训练
    for (int i = 0; i < 100; ++i) {
        x->set_input(i * 0.1);
        loss->zero_grad_recursive();
        loss->calc();
        loss->backward();
        
        for (auto& param : graph.parameter_nodes) {
            param->update(0.01);
        }
    }
    
    // 保存参数
    graph.SaveParams("model_params.txt");
    std::cout << "Model saved to model_params.txt" << std::endl;
    
    // 重置参数
    for (auto& param : graph.parameter_nodes) {
        for (size_t i = 0; i < param->size(); ++i) {
            param->Item(i) = 0.0;
        }
    }
    std::cout << "Parameters reset to zero" << std::endl;
    
    // 加载参数
    graph.LoadParams("model_params.txt");
    std::cout << "Parameters loaded from model_params.txt" << std::endl;
    
    // 测试加载的模型
    x->set_input(1.0);
    y->calc();
    std::cout << "Prediction with loaded model: " << y->item() << std::endl;
    
    return 0;
}