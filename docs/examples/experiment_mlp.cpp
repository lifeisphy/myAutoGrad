#include "../../autograd.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

int main() {
    // 网络参数
    const int input_size = 784;  // 28x28
    const int hidden_size = 128;
    const int output_size = 10;
    
    // 创建变量
    auto x = make_input(std::vector<double>(input_size, 0.0), {input_size});
    auto target = make_input(std::vector<double>(output_size, 0.0), {output_size});
    
    // 网络层
    auto W1 = make_param(vec_r(input_size * hidden_size), {hidden_size, input_size});
    auto b1 = make_param(vec_r(hidden_size), {hidden_size});
    auto W2 = make_param(vec_r(hidden_size * output_size), {output_size, hidden_size});
    auto b2 = make_param(vec_r(output_size), {output_size});
    
    // 构建网络
    auto z1 = add(mul(W1, x, 1, 0), b1);
    auto a1 = relu(z1);
    auto z2 = add(mul(W2, a1, 1, 0), b2);
    auto output = z2;
    auto loss = mse_loss(output, target);
    
    // 创建计算图和优化器
    auto graph = ComputationGraph::BuildFromOutput(loss);
    AdamOptimizer optimizer(0.001);
    optimizer.set_parameter_nodes(graph.parameter_nodes);
    
    // 模拟训练数据
    std::vector<std::vector<double>> train_images(100, std::vector<double>(input_size));
    std::vector<std::vector<double>> train_labels(100, std::vector<double>(output_size, 0.0));
    
    std::random_device rd;
    std::mt19937 gen(42);  // 固定随机种子以确保结果可重现
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    // 生成随机数据
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < input_size; ++j) {
            train_images[i][j] = dis(gen);
        }
        int label = i % 10;
        train_labels[i][label] = 1.0;
    }
    
    // 训练循环
    const int num_epochs = 10;
    std::cout << "=== MLP Experiment for Report ===" << std::endl;
    std::cout << "Network Architecture:" << std::endl;
    std::cout << "Input: " << input_size << " neurons" << std::endl;
    std::cout << "Hidden: " << hidden_size << " neurons" << std::endl;
    std::cout << "Output: " << output_size << " neurons" << std::endl;
    std::cout << "Parameters: " << (input_size * hidden_size + hidden_size + output_size) << std::endl;
    std::cout << std::endl;
    
    // 在循环外定义变量，以便在循环结束后访问
    double total_loss = 0.0;
    int correct = 0;
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        total_loss = 0.0;  // 重置每个epoch的损失
        correct = 0;       // 重置每个epoch的准确率
        
        for (int i = 0; i < 100; ++i) {
            x->set_input(train_images[i]);
            target->set_input(train_labels[i]);
            
            loss->zero_grad_recursive();
            loss->calc();
            loss->backward();
            
            // 更新参数
            for (auto& param : graph.parameter_nodes) {
                optimizer.update(param);
            }
            
            total_loss += loss->item();
            
            // 计算准确率
            output->calc();
            int predicted = 0;
            double max_val = output->data()[0];
            for (int j = 1; j < output_size; ++j) {
                if (output->data()[j] > max_val) {
                    max_val = output->data()[j];
                    predicted = j;
                }
            }
            
            int actual = i % 10;
            if (predicted == actual) correct++;
            
            if (i % 10 == 0) {
                std::cout << "Epoch " << epoch + 1 << ", Processed " << i << "/100 samples" << std::endl;
            }
        }
        
        std::cout << "Epoch " << epoch + 1 << " completed, Loss: " << std::fixed << std::setprecision(6)
                  << total_loss / 100 << ", Accuracy: " << correct << "/100" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Final Results:" << std::endl;
    std::cout << "Final Training Loss: " << std::fixed << std::setprecision(6) << total_loss / 100 << std::endl;
    std::cout << "Final Test Accuracy: " << std::fixed << std::setprecision(1) << correct << "/100" << std::endl;
    std::cout << "Total Parameters: " << (input_size * hidden_size + hidden_size + output_size) << std::endl;
    
    return 0;
}