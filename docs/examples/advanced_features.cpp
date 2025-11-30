#include "../../autograd.hpp"
#include <iostream>

// 计算图可视化
int graph_visualization_example() {
    // 构建模型...
    auto loss = build_model();
    auto graph = ComputationGraph::BuildFromOutput(loss);
    
    // 可视化计算图
    graph->Visualize("graph.dot");
    
    // 使用 Graphviz 渲染
    // dot -Tpng graph.dot -o graph.png
    
    return 0;
}

// 参数保存和加载
int parameter_save_load_example() {
    // 构建模型...
    auto loss = build_model();
    auto graph = ComputationGraph::BuildFromOutput(loss);
    
    // 训练...
    
    // 保存参数
    graph->SaveParams("model_params.txt");
    
    // 加载参数
    graph->LoadParams("model_params.txt");
    
    return 0;
}

// 使用宏简化代码
int macro_example() {
    // 使用宏创建变量
    MAKE_INPUT(x, std::vector<double>{1.0, 2.0, 3.0}, {3});
    MAKE_PARAM(w, std::vector<double>{0.5, 0.5, 0.5}, {3});
    MAKE_PARAM(b, 0.1);
    
    // 使用宏构建计算图
    MUL(wx, w, x);
    ADD(output, wx, b);
    
    // 前向计算
    output->calc();
    
    std::cout << "Output: " << output->item() << std::endl;
    
    return 0;
}

int main() {
    std::cout << "Advanced features examples:" << std::endl;
    std::cout << "1. Graph visualization" << std::endl;
    std::cout << "2. Parameter save/load" << std::endl;
    std::cout << "3. Macro usage" << std::endl;
    
    return 0;
}