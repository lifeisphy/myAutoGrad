#include "../../autograd.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>

int main() {
    // 网络参数
    const int input_dim = 10;       // 输入维度（前10个值）
    const int hidden_dim = 32;      // 隐藏层维度
    const int output_dim = 1;       // 输出维度 (单个值)
    const int seq_length = 20;      // 序列长度
    const double frequency = 15.0;  // sin(ax)中的频率a
    
    std::cout << "=== RNN Sine Function Learning Experiment (Debug Version) ===" << std::endl;
    std::cout << "Network Architecture:" << std::endl;
    std::cout << "Input Dimension: " << input_dim << " (previous 10 values)" << std::endl;
    std::cout << "Hidden Dimension: " << hidden_dim << std::endl;
    std::cout << "Output Dimension: " << output_dim << " (single value)" << std::endl;
    std::cout << "Sequence Length: " << seq_length << std::endl;
    std::cout << "Target Function: sin(" << frequency << "x)" << std::endl;
    std::cout << std::endl;
    
    // 创建变量 - 为每个时间步创建独立的输入变量
    auto hidden = make_input(vec_r(hidden_dim, 0.0), {hidden_dim}, "hidden_0");
    std::vector<VarPtr> inputs;
    for (int t = 0; t < seq_length; ++t) {
        inputs.push_back(make_input(vec_r(input_dim, 0.0), {input_dim}, "input_" + std::to_string(t)));
    }
    auto target = make_input(vec_r(output_dim, 0.0), {output_dim}, "target");
    
    // 手动构建简单的RNN，避免使用RecurrentOperation类
    // 创建参数
    auto W_hh = make_param(vec_r(hidden_dim * hidden_dim, 0.1), {hidden_dim, hidden_dim});  // hidden-to-hidden
    auto W_xh = make_param(vec_r(input_dim * hidden_dim, 0.1), {input_dim, hidden_dim});  // input-to-hidden
    auto b_h = make_param(vec_r(hidden_dim, 0.0), {hidden_dim});  // hidden bias
    auto W_hy = make_param(vec_r(hidden_dim * output_dim, 0.1), {hidden_dim, output_dim});  // hidden-to-output
    auto b_y = make_param(vec_r(output_dim, 0.0), {output_dim});  // output bias
    
    // 构建RNN序列
    std::vector<VarPtr> hidden_states;
    std::vector<VarPtr> outputs;
    
    VarPtr h_prev = hidden;
    
    for (int t = 0; t < seq_length; ++t) {
        // RNN计算: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
        auto hh_term = mul(W_hh, h_prev, 0, 0);  // hidden-to-hidden: [hidden_dim, hidden_dim] * [hidden_dim]
        auto xh_term = mul(W_xh, inputs[t], 0, 0);   // input-to-hidden: [input_dim, hidden_dim] * [input_dim]
        auto sum1 = add(hh_term, xh_term);
        auto h_t = tanh_activation(add(sum1, b_h), "h_" + std::to_string(t));
        
        // 输出计算: y_t = W_hy * h_t + b_y
        auto y_t = add(mul(W_hy, h_t, 0, 0), b_y, "y_" + std::to_string(t));  // hidden-to-output: [hidden_dim, output_dim] * [hidden_dim]
        
        hidden_states.push_back(h_t);
        outputs.push_back(y_t);
        h_prev = h_t;
    }
    
    // 使用最后一个时间步的输出作为预测
    auto final_output = outputs.back();
    auto loss = mse_loss(final_output, target);
    
    // 创建计算图和优化器
    auto graph = ComputationGraph::BuildFromOutput(loss);
    AdamOptimizer optimizer(0.01, 0.9, 0.999, 1e-8);  // 使用标准的Adam参数
    optimizer.set_parameter_nodes(graph.parameter_nodes);
    
    // 计算总参数数量
    size_t total_params = 0;
    for (const auto& param : graph.parameter_nodes) {
        total_params += param->size();
    }
    std::cout << "Total Parameters: " << total_params << std::endl;
    std::cout << std::endl;
    
    // 生成正弦函数训练数据
    std::vector<std::vector<double>> train_sequences(200, std::vector<double>(seq_length + input_dim));
    std::vector<double> train_targets(200);
    
    std::random_device rd;
    std::mt19937 gen(42);  // 固定随机种子以确保结果可重现
    std::uniform_real_distribution<> dis_start(0.0, 1.0);  // 0到1的随机起始点
    
    // 生成正弦函数序列数据
    for (int i = 0; i < 200; ++i) {
        double start_x = dis_start(gen);
        std::vector<double> sequence(seq_length + input_dim);
        for (int t = 0; t < seq_length + input_dim; ++t) {
            double x = start_x + t * 0.01;  // 步长为0.01
            sequence[t] = std::sin(frequency * x);
        }
        train_sequences[i] = sequence;
        // 目标是序列的下一个值
        double next_x = start_x + (seq_length + input_dim) * 0.01;
        train_targets[i] = std::sin(frequency * next_x);
    }
    
    // 检查数据范围
    std::cout << "Data range check:" << std::endl;
    std::cout << "Min sequence value: " << *std::min_element(train_sequences[0].begin(), train_sequences[0].end()) << std::endl;
    std::cout << "Max sequence value: " << *std::max_element(train_sequences[0].begin(), train_sequences[0].end()) << std::endl;
    std::cout << "Min target value: " << *std::min_element(train_targets.begin(), train_targets.end()) << std::endl;
    std::cout << "Max target value: " << *std::max_element(train_targets.begin(), train_targets.end()) << std::endl;
    std::cout << std::endl;
    
    // 训练循环
    const int num_epochs = 50;  // 增加epoch数量到50
    
    // 在循环外定义变量，以便在循环结束后访问
    double total_loss = 0.0;
    double total_error = 0.0;
    
    // 记录每个epoch的损失和误差用于可视化
    std::vector<double> epoch_losses;
    std::vector<double> epoch_errors;
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        total_loss = 0.0;  // 重置每个epoch的损失
        total_error = 0.0; // 重置每个epoch的误差
        
        for (int i = 0; i < 100; ++i) {  // 只处理100个样本进行调试
            // 重置隐藏状态
            hidden->set_input(vec_r(hidden_dim, 0.0));
            
            // 逐个时间步处理序列
            for (int t = 0; t < seq_length; ++t) {
                // 设置当前时间步的输入 - 使用前input_dim个值
                std::vector<double> x_val(input_dim);
                for (int d = 0; d < input_dim; ++d) {
                    // 从第10个点开始作为输入，避免0填充
                    x_val[d] = train_sequences[i][t + d];
                }
                
                // 调试：输出第一个样本的前几个时间步的输入
                if (i == 0 && (t == 0 || t == 1 || t == 2)) {
                    std::cout << "Sample " << i << ", Time step " << t << ", Input values: ";
                    for (int d = 0; d < input_dim; ++d) {
                        std::cout << x_val[d] << " ";
                    }
                    std::cout << std::endl;
                }
                
                // 设置对应时间步的输入变量
                inputs[t]->set_input(x_val);
                
                // 计算当前时间步的输出
                hidden_states[t]->calc();
                outputs[t]->calc();
            }
            
            // 设置目标
            std::vector<double> target_val = {train_targets[i]};
            target->set_input(target_val);
            
            // 计算最终输出和损失
            final_output->calc();
            loss->calc();
            
            // 反向传播
            loss->zero_grad_recursive();
            loss->backward();
            
            // 更新参数
            for (auto& param : graph.parameter_nodes) {
                optimizer.update(param);
            }
            
            total_loss += loss->item();
            
            // 计算预测误差
            double predicted = final_output->data()[0];
            double actual = train_targets[i];
            double error = std::abs(predicted - actual);
            total_error += error;
            
            // 调试：输出前几个样本的详细信息
            if (i < 5) {
                std::cout << "Sample " << i << ": Predicted=" << predicted << ", Actual=" << actual 
                          << ", Error=" << error << ", Loss=" << loss->item() << std::endl;
            }
            
            if (i % 20 == 0) {
                std::cout << "Epoch " << epoch + 1 << ", Processed " << i << "/100 samples" << std::endl;
            }
        }
        
        std::cout << "Epoch " << epoch + 1 << " completed, Loss: " << std::fixed << std::setprecision(6)
                  << total_loss / 100 << ", Avg Error: " << std::setprecision(6) << total_error / 100 << std::endl;
        
        // 记录当前epoch的损失和误差
        epoch_losses.push_back(total_loss / 100);
        epoch_errors.push_back(total_error / 100);
    }
    
    // 测试阶段 - 使用20个样本进行测试，在[0,1]均匀分布的数据点上
    std::cout << std::endl;
    std::cout << "Final Testing (20 samples on [0,1] uniform distribution):" << std::endl;
    
    double test_total_error = 0.0;
    
    // 生成20个在[0,1]均匀分布的测试点
    std::vector<double> test_points(20);
    for (int i = 0; i < 20; ++i) {
        test_points[i] = i / 19.0;  // 从0到1的均匀分布
    }
    
    for (int i = 0; i < 20; ++i) {
        // 重置隐藏状态
        hidden->set_input(vec_r(hidden_dim, 0.0));
        
        // 为当前测试点生成序列
        double start_x = test_points[i];
        std::vector<double> test_sequence(seq_length + input_dim);
        for (int t = 0; t < seq_length + input_dim; ++t) {
            double x = start_x + t * 0.01;  // 步长为0.01
            test_sequence[t] = std::sin(frequency * x);
        }
        
        // 逐个时间步处理序列
        for (int t = 0; t < seq_length; ++t) {
            // 设置当前时间步的输入 - 使用前input_dim个值
            std::vector<double> x_val(input_dim);
            for (int d = 0; d < input_dim; ++d) {
                x_val[d] = test_sequence[t + d];
            }
            
            // 设置对应时间步的输入变量
            inputs[t]->set_input(x_val);
            
            // 计算当前时间步的输出
            hidden_states[t]->calc();
            outputs[t]->calc();
        }
        
        // 设置目标 - 序列的下一个值
        double next_x = start_x + (seq_length + input_dim) * 0.01;
        std::vector<double> target_val = {std::sin(frequency * next_x)};
        target->set_input(target_val);
        
        // 计算最终输出和损失
        final_output->calc();
        loss->calc();
        
        // 反向传播（测试阶段也需要）
        loss->zero_grad_recursive();
        loss->backward();
        
        // 获取预测结果
        double predicted = final_output->data()[0];
        double actual = target_val[0];
        double error = std::abs(predicted - actual);
        test_total_error += error;
        
        std::cout << "Test Point " << i << " (x=" << std::fixed << std::setprecision(3) << start_x
                  << "): Predicted=" << std::setprecision(6) << predicted
                  << ", Actual=" << actual << ", Error=" << error << std::endl;
    }
    
    // 保存测试结果到CSV文件
    std::ofstream csv_file("rnn_uniform_test_predictions.csv");
    csv_file << "TestPoint,InputX,Predicted,Actual,Error\n";
    for (int i = 0; i < 20; ++i) {
        // 重置隐藏状态
        hidden->set_input(vec_r(hidden_dim, 0.0));
        
        // 为当前测试点生成序列
        double start_x = test_points[i];
        std::vector<double> test_sequence(seq_length + input_dim);
        for (int t = 0; t < seq_length + input_dim; ++t) {
            double x = start_x + t * 0.01;  // 步长为0.01
            test_sequence[t] = std::sin(frequency * x);
        }
        
        // 逐个时间步处理序列
        for (int t = 0; t < seq_length; ++t) {
            // 设置当前时间步的输入 - 使用前input_dim个值
            std::vector<double> x_val(input_dim);
            for (int d = 0; d < input_dim; ++d) {
                x_val[d] = test_sequence[t + d];
            }
            
            // 设置对应时间步的输入变量
            inputs[t]->set_input(x_val);
            
            // 计算当前时间步的输出
            hidden_states[t]->calc();
            outputs[t]->calc();
        }
        
        // 设置目标 - 序列的下一个值
        double next_x = start_x + (seq_length + input_dim) * 0.01;
        std::vector<double> target_val = {std::sin(frequency * next_x)};
        target->set_input(target_val);
        
        // 计算最终输出和损失
        final_output->calc();
        loss->calc();
        
        // 反向传播（测试阶段也需要）
        loss->zero_grad_recursive();
        loss->backward();
        
        // 获取预测结果
        double predicted_val = final_output->data()[0];
        double actual_val = target_val[0];
        double error_val = std::abs(predicted_val - actual_val);
        
        // 写入CSV文件
        csv_file << i << "," << start_x << "," << predicted_val << "," << actual_val << "," << error_val << "\n";
    }
    csv_file.close();
    
    std::cout << std::endl;
    std::cout << "Final Results:" << std::endl;
    std::cout << "Final Training Loss: " << std::fixed << std::setprecision(6) << total_loss / 100 << std::endl;
    std::cout << "Final Average Error: " << std::setprecision(6) << total_error / 100 << std::endl;
    std::cout << "Test Average Error (20 uniform points): " << std::setprecision(6) << test_total_error / 20 << std::endl;
    std::cout << "Total Parameters: " << total_params << std::endl;
    std::cout << "Test results saved to rnn_uniform_test_predictions.csv" << std::endl;
    
    // 保存训练过程数据到CSV文件
    std::ofstream training_csv_file("rnn_training_progress.csv");
    training_csv_file << "Epoch,Loss,AverageError\n";
    for (int i = 0; i < num_epochs; ++i) {
        training_csv_file << (i + 1) << "," << epoch_losses[i] << "," << epoch_errors[i] << "\n";
    }
    training_csv_file.close();
    std::cout << "Training progress saved to rnn_training_progress.csv" << std::endl;
    
    return 0;
}