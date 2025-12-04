#include "../../autograd.hpp"
#include <iostream>
#include <vector>

// 使用 Adam 优化器
int adam_example() {
    // 构建模型...
    auto loss = build_model();
    auto graph = ComputationGraph::BuildFromOutput(loss);
    
    // 创建优化器
    AdamOptimizer optimizer(0.001);  // 学习率 0.001
    optimizer.set_parameter_nodes(graph.parameter_nodes);
    
    // 训练循环
    for (int epoch = 0; epoch < 100; ++epoch) {
        // 加载数据
        load_data(graph);
        
        // 前向传播
        loss->calc();
        
        // 反向传播
        loss->backward();
        
        // 使用优化器更新参数
        for (auto& param : graph.parameter_nodes) {
            optimizer.update(param);
        }
        
        std::cout << "Epoch " << epoch << ", Loss: " << loss->item() << std::endl;
    }
    
    return 0;
}

// 使用 fit 方法
int fit_example() {
    // 构建模型...
    auto loss = build_model();
    auto graph = ComputationGraph::BuildFromOutput(loss);
    
    // 定义数据加载函数
    auto load_data = [&](ComputationGraph* g) {
        // 加载第 g->i 个样本
        auto x = g->get_node_by_name("x");
        auto y = g->get_node_by_name("target");
        
        x->set_input(get_input_data(g->i));
        y->set_input(get_target_data(g->i));
    };
    
    // 定义打印函数
    auto print_info = [&](ComputationGraph* g) {
        std::cout << "Sample " << g->i << ", Loss: " << g->output_nodes[0]->item() << std::endl;
    };
    
    // 使用 fit 方法训练
    graph->fit(load_data, 10, 1000, 0.001, nullptr, print_info);
    
    return 0;
}

int main() {
    std::cout << "Model training examples:" << std::endl;
    std::cout << "1. Adam optimizer example" << std::endl;
    std::cout << "2. Fit method example" << std::endl;
    
    return 0;
}