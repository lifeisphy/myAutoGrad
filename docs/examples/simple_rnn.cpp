#include "../../autograd.hpp"
#include <iostream>
#include <vector>

int main() {
    // 参数
    const int input_size = 3;
    const int hidden_size = 4;
    const int seq_length = 5;
    
    // 创建变量
    auto x = make_input(std::vector<double>(input_size, 0.0), {input_size});
    auto h_prev = make_input(std::vector<double>(hidden_size, 0.0), {hidden_size});
    
    // 创建 RNN 操作 - 直接使用线性变换而不是linear_函数
    auto rnn_op = [](VarPtr hidden_state, VarPtr input, bool make_params, std::vector<VarPtr>& params) -> VarPtr {
        size_t hidden_dim = hidden_state->size(); // 使用实际的hidden_state大小
        size_t input_dim = input->size();
        
        if(make_params){
            // 创建参数
            auto W_h = make_param(vec_r(hidden_dim * hidden_dim), {hidden_dim, hidden_dim});
            auto W_x = make_param(vec_r(hidden_dim * input_dim), {hidden_dim, input_dim});
            auto b = make_param(vec_r(hidden_dim), {hidden_dim});
            params.push_back(W_h);
            params.push_back(W_x);
            params.push_back(b);
        }
        
        // 线性变换: h_new = W_h * h_prev + W_x * x + b
        auto h_term = mul(params[0], hidden_state, 1, 0); // W_h * h_prev
        auto x_term = mul(params[1], input, 1, 0); // W_x * x
        auto h_new = add(add(h_term, x_term), params[2]); // W_h * h_prev + W_x * x + b
        
        return h_new;
    };
    auto rnn = RecurrentOperation(rnn_op, h_prev, x);
    
    // 展开循环
    rnn.expand(seq_length);
    
    // 获取输出
    auto outputs = rnn.outputs;
    auto final_hidden = rnn.hidden.back();
    
    // 创建目标
    auto target = make_input(std::vector<double>(hidden_size, 0.0), {hidden_size});
    auto loss = mse_loss(final_hidden, target);
    
    // 创建计算图
    auto graph = ComputationGraph::BuildFromOutput(loss);
    
    // 模拟序列数据
    std::vector<std::vector<double>> sequences = {
        {1.0, 2.0, 3.0},
        {0.5, 1.5, 2.5},
        {2.0, 1.0, 0.5},
        {1.5, 2.5, 3.5},
        {0.0, 1.0, 2.0}
    };
    
    // 训练
    double learning_rate = 0.01;
    for (int epoch = 0; epoch < 100; ++epoch) {
        double total_loss = 0.0;
        
        // 设置初始隐藏状态
        h_prev->set_input(std::vector<double>(hidden_size, 0.0));
        
        // 设置输入序列
        for (int t = 0; t < seq_length; ++t) {
            if (t == 0) {
                x->set_input(sequences[t]);
            } else {
                // 对于后续时间步，需要设置对应的输入节点
                if (t < rnn.inputs.size()) {
                    auto input_node = rnn.inputs[t];
                    input_node->set_input(sequences[t]);
                }
            }
        }
        
        // 设置目标
        target->set_input(std::vector<double>(hidden_size, 0.5));
        
        // 前向和反向传播
        loss->zero_grad_recursive();
        loss->calc();
        loss->backward();
        
        // 更新参数
        for (auto& param : graph.parameter_nodes) {
            param->update(learning_rate);
        }
        
        total_loss += loss->item();
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss << std::endl;
        }
    }
    
    return 0;
}