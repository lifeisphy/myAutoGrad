#include "../../autograd.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

int main() {
    // 网络参数
    const int n = 28;           // 图像大小 28x28
    const int n_input = n * n; // 784
    const int n_output = 10;    // 10个类别
    const int n_kernel = 16;    // 减少卷积核数量以简化实验
    const int n_kernel_2 = 32;  // 第二层卷积核数量
    
    std::cout << "=== CNN Experiment for Report ===" << std::endl;
    std::cout << "Network Architecture:" << std::endl;
    std::cout << "Input: " << n << "x" << n << " images" << std::endl;
    std::cout << "Conv1: " << n_kernel << " kernels, 3x3" << std::endl;
    std::cout << "Conv2: " << n_kernel_2 << " kernels, 3x3" << std::endl;
    std::cout << "Output: " << n_output << " classes" << std::endl;
    std::cout << std::endl;
    
    // 创建变量
    MAKE_INPUT(x, std::vector<double>(n_input, 0.0), {n, n});
    MAKE_INPUT(label, std::vector<double>(n_output, 0.0), {n_output});
    
    // 初始化权重
    std::random_device rd;
    std::mt19937 gen(42);  // 固定随机种子以确保结果可重现
    std::normal_distribution<double> normal_dist(0.0, 0.1);
    
    auto init_weights = [&](size_t size) {
        std::vector<double> weights(size);
        for (size_t i = 0; i < size; i++) {
            weights[i] = normal_dist(gen);
        }
        return weights;
    };
    
    // 定义网络参数
    MAKE_PARAM(kernel_1, init_weights(3 * 3 * n_kernel), {3, 3, n_kernel});
    MAKE_PARAM(kernel_2, init_weights(3 * 3 * n_kernel_2 * n_kernel), {3, 3, n_kernel, n_kernel_2});
    
    // 第一个卷积层
    std::vector<VarPtr> output_1;
    for (int i = 0; i < n_kernel; i++) {
        SLICE(kernel_slice, kernel_1, {-1, -1, i});
        CONV2D(conv_out, x, kernel_slice);
        RELU(relu_out, conv_out);
        MAXPOOLING(pool_out, relu_out, 2);
        output_1.push_back(pool_out);
    }
    STACK(feature_maps_1, output_1);  // [16, 13, 13]
    
    // 第二个卷积层
    std::vector<VarPtr> slices_1;
    for (int i = 0; i < n_kernel; i++) {
        slices_1.push_back(slice(feature_maps_1, {i, -1, -1}));
    }
    
    std::vector<VarPtr> output_2;
    for (int i = 0; i < n_kernel_2; i++) {
        std::vector<VarPtr> conv_results;
        for (int j = 0; j < n_kernel; j++) {
            SLICE(kernel_slice, kernel_2, {-1, -1, j, i});
            CONV2D(conv_out, slices_1[j], kernel_slice);
            RELU(relu_out, conv_out);
            MAXPOOLING(pool_out, relu_out, 2);
            conv_results.push_back(pool_out);
        }
        SUM(summed, conv_results);
        output_2.push_back(summed);
    }
    STACK(feature_maps_2, output_2);  // [32, 5, 5]
    
    // 全连接层
    size_t input_size = feature_maps_2->size();
    const int mid_size = 128;
    
    MAKE_PARAM(W1, init_weights(input_size * mid_size), {input_size, mid_size});
    MAKE_PARAM(b1, init_weights(mid_size), {mid_size});
    MAKE_PARAM(W2, init_weights(mid_size * n_output), {mid_size, n_output});
    MAKE_PARAM(b2, init_weights(n_output), {n_output});
    
    FLATTEN(flattened, feature_maps_2);
    MUL(mul1, W1, flattened, 0, 0);
    ADD(layer1_pre, mul1, b1);
    RELU(layer1, layer1_pre);
    MUL(mul2, W2, layer1, 0, 0);
    ADD(output, mul2, b2);
    MSE_LOSS(loss, output, label);
    
    // 创建计算图和优化器
    auto graph = ComputationGraph::BuildFromOutput(loss);
    AdamOptimizer optimizer(0.001);
    optimizer.set_parameter_nodes(graph.parameter_nodes);
    
    // 计算总参数数量
    size_t total_params = 0;
    for (const auto& param : graph.parameter_nodes) {
        total_params += param->size();
    }
    std::cout << "Total Parameters: " << total_params << std::endl;
    std::cout << std::endl;
    
    // 模拟训练数据
    std::vector<std::vector<double>> train_images(100, std::vector<double>(n_input));
    std::vector<std::vector<double>> train_labels(100, std::vector<double>(n_output, 0.0));
    
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    // 生成随机数据
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < n_input; ++j) {
            train_images[i][j] = dis(gen);
        }
        int label_idx = i % 10;
        train_labels[i][label_idx] = 1.0;
    }
    
    // 训练循环
    const int num_epochs = 5;  // CNN训练更慢，减少epoch数
    
    // 在循环外定义变量，以便在循环结束后访问
    double total_loss = 0.0;
    int correct = 0;
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        total_loss = 0.0;  // 重置每个epoch的损失
        correct = 0;       // 重置每个epoch的准确率
        
        for (int i = 0; i < 100; ++i) {
            x->set_input(train_images[i]);
            label->set_input(train_labels[i]);
            
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
            for (int j = 1; j < n_output; ++j) {
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
    std::cout << "Total Parameters: " << total_params << std::endl;
    
    return 0;
}