#include "../../autograd.hpp"
#include <iostream>

int main() {
    // 构建复杂的计算图
    auto x = make_input(2.0);
    auto w1 = make_param(0.5);
    auto w2 = make_param(0.3);
    auto b = make_param(0.1);
    
    auto h1 = relu(add(mul(w1, x), b));
    auto h2 = sigmoid(add(mul(w2, x), b));
    auto output = add(h1, h2);
    auto loss = mse_loss(output, make_var(1.0));
    
    // 创建计算图
    auto graph = ComputationGraph::BuildFromOutput(loss);
    
    // 可视化
    graph.Visualize("example_graph.dot");
    
    // 打印摘要
    graph.print_summary();
    
    std::cout << "Graph visualization saved to example_graph.dot" << std::endl;
    std::cout << "Use 'dot -Tpng example_graph.dot -o example_graph.png' to render" << std::endl;
    
    return 0;
}