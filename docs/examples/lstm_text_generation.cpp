#include "../../autograd.hpp"
#include <iostream>
#include <vector>
#include <map>

int main() {
    // 简化的字符级语言模型
    const int vocab_size = 10;  // 简化词汇表
    const int hidden_size = 8;
    const int seq_length = 4;
    
    // 创建变量
    auto x = make_input(std::vector<double>(vocab_size, 0.0), {vocab_size});
    auto hidden_state = make_input(std::vector<double>(hidden_size * 2, 0.0), {hidden_size * 2});
    
    // 创建 LSTM
    auto lstm_op = lstm_(hidden_size, hidden_size);
    auto lstm = RecurrentOperation(lstm_op, hidden_state, x);
    
    // 展开循环
    lstm.expand(seq_length);
    
    // 输出层
    auto final_hidden = lstm.hidden.back();
    // 从LSTM的隐藏状态中提取h部分（前hidden_size个元素）
    auto h_final = slice_indices(final_hidden, {idx_range(0, hidden_size, 1)}, "h_final");
    auto W_out = make_param(vec_r(hidden_size * vocab_size), {vocab_size, hidden_size});
    auto b_out = make_param(vec_r(vocab_size), {vocab_size});
    auto logits = add(mul(W_out, h_final, 1, 0), b_out);
    
    // 目标
    auto target = make_input(std::vector<double>(vocab_size, 0.0), {vocab_size});
    auto loss = mse_loss(logits, target);
    
    // 创建计算图
    auto graph = ComputationGraph::BuildFromOutput(loss);
    
    // 模拟训练数据
    std::vector<std::vector<double>> sequences = {
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // 字符 0
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},  // 字符 1
        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},  // 字符 2
        {0, 0, 0, 1, 0, 0, 0, 0, 0, 0}   // 字符 3
    };
    
    // 训练
    double learning_rate = 0.01;
    for (int epoch = 0; epoch < 50; ++epoch) {
        double total_loss = 0.0;
        
        // 设置初始状态
        hidden_state->set_input(std::vector<double>(hidden_size * 2, 0.0));
        
        // 设置输入序列
        for (int t = 0; t < seq_length; ++t) {
            if (t == 0) {
                x->set_input(sequences[t]);
            } else {
                auto input_node = lstm.inputs[t];
                input_node->set_input(sequences[t]);
            }
        }
        
        // 设置目标（预测下一个字符）
        target->set_input(sequences[0]);  // 简化：总是预测第一个字符
        
        // 训练步骤
        loss->zero_grad_recursive();
        loss->calc();
        loss->backward();
        
        for (auto& param : graph.parameter_nodes) {
            param->update(learning_rate);
        }
        
        total_loss += loss->item();
        
        if (epoch % 5 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss << std::endl;
        }
    }
    
    // 生成文本
    std::cout << "\nGenerating text:" << std::endl;
    hidden_state->set_input(std::vector<double>(hidden_size * 2, 0.0));
    x->set_input(sequences[0]);
    
    for (int i = 0; i < 10; ++i) {
        logits->calc();
        
        // 找到概率最高的字符
        int predicted = 0;
        double max_val = logits->data()[0];
        for (int j = 1; j < vocab_size; ++j) {
            if (logits->data()[j] > max_val) {
                max_val = logits->data()[j];
                predicted = j;
            }
        }
        
        std::cout << "Char " << predicted << " ";
        
        // 使用预测作为下一个输入
        std::vector<double> next_input(vocab_size, 0.0);
        next_input[predicted] = 1.0;
        x->set_input(next_input);
    }
    std::cout << std::endl;
    
    return 0;
}