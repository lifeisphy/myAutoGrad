// mnist_train.cpp
#include "../autograd.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <signal.h>

ComputationGraph* pgraph = nullptr; // Global pointer to the computation graph
void signal_handler(int signal){
    if(signal == SIGINT){
        std::cout << "\nTraining interrupted by user." << std::endl;
        pgraph->SaveParams("test/mnist_model_params_interrupt.txt");
        exit(0);
    }
}

class MNISTDataset {
private:
    std::vector<std::vector<double>> images_;
    std::vector<int> labels_;
    size_t size_;

public:
    bool load_csv(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        // 跳过表头
        std::getline(file, line);
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string item;
            
            // 读取标签
            std::getline(ss, item, ',');
            int label = std::stoi(item);
            labels_.push_back(label);
            
            // 读取像素值
            std::vector<double> pixels;
            while (std::getline(ss, item, ',')) {
                pixels.push_back(std::stod(item) / 255.0);  // 归一化到[0,1]
            }
            images_.push_back(pixels);
        }
        
        size_ = labels_.size();
        std::cout << "Loaded " << size_ << " samples" << std::endl;
        return true;
    }
    
    size_t size() const { return size_; }
    
    const std::vector<double>& get_image(size_t idx) const {
        return images_[idx];
    }
    
    int get_label(size_t idx) const {
        return labels_[idx];
    }
    
    std::vector<double> get_one_hot_label(size_t idx, int num_classes = 10) const {
        std::vector<double> one_hot(num_classes, 0.0);
        one_hot[labels_[idx]] = 1.0;
        return one_hot;
    }
};

int main(int argc, char* argv[]) {
    std::vector<bool> results;
    std::cout << "=== MNIST CNN Training in C++ ===" << std::endl;
    
    // 加载数据
    MNISTDataset dataset;
    if (!dataset.load_csv("testcases/digit-recognizer/train.csv")) {
        return -1;
    }
    
    // 网络参数
    const int n = 28;
    const int n_input = n * n;
    const int n_output = 10;
    const int n_kernel = 32;
    const int n_kernel_2 = 48;
    
    // 创建网络变量
    auto x = make_input(std::vector<double>(n_input, 0.0), {n, n});
    auto label = make_input(std::vector<double>(n_output, 0.0), {n_output});
    
    // 初始化卷积核权重
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
    
    auto kernel_1 = make_param(init_weights(3 * 3 * n_kernel), {3, 3, n_kernel});
    auto kernel_2 = make_param(init_weights(3 * 3 * n_kernel_2 * n_kernel), {3, 3, n_kernel, n_kernel_2});
    
    // 第一个卷积层
    std::cout << "Building first conv layer..." << std::endl;
    std::vector<VarPtr> output_1;
    for (int i = 0; i < n_kernel; i++) {
        auto kernel_slice = slice(kernel_1, {-1, -1, i});  // 提取第i个卷积核
        auto conv_out = conv2d(x, kernel_slice);
        auto relu_out = sigmoid(conv_out);
        auto pool_out = MaxPooling(relu_out, 2);
        output_1.push_back(pool_out);
    }
    auto feature_maps_1 = stack(output_1);  // [32, 13, 13]
    std::cout<<"Feature maps 1 shape: " << feature_maps_1->shape()[0] << "x" << feature_maps_1->shape()[1] << "x" << feature_maps_1->shape()[2] << std::endl;

    // 第二个卷积层
    std::cout << "Building second conv layer..." << std::endl;
    std::vector<VarPtr> slices_1;
    for (int i = 0; i < n_kernel; i++) {
        slices_1.push_back(slice(feature_maps_1, {i, -1, -1}));
    }
    
    std::vector<VarPtr> output_2;
    for (int i = 0; i < n_kernel_2; i++) {
        std::vector<VarPtr> conv_results;
        for (int j = 0; j < n_kernel; j++) {
            auto kernel_slice = slice(kernel_2, {-1, -1, j, i});
            auto conv_out = conv2d(slices_1[j], kernel_slice);
            auto relu_out = sigmoid(conv_out);
            auto pool_out = MaxPooling(relu_out, 2);
            conv_results.push_back(pool_out);
        }
        auto summed = sum(conv_results);
        output_2.push_back(summed);
    }
    auto feature_maps_2 = stack(output_2);
    std::cout<<"Feature maps 2 shape: " << feature_maps_2->shape()[0] << "x" << feature_maps_2->shape()[1] << "x" << feature_maps_2->shape()[2] << std::endl;
    // 全连接层
    std::cout << "Building fully connected layers..." << std::endl;
    size_t input_size = feature_maps_2->size();
    const int mid_size = 128;
    std::cout<<"input size: "<< input_size <<std::endl;
    auto W1 = make_param(init_weights(input_size * mid_size), {input_size, mid_size});
    auto b1 = make_param(init_weights(mid_size), {mid_size});
    auto W2 = make_param(init_weights(mid_size * n_output), {mid_size, n_output});
    auto b2 = make_param(init_weights(n_output), {n_output});
    std::cout<<"W1 shape: "<< W1->shape()[0] << "x" << W1->shape()[1] <<std::endl;
    std::cout<<"W2 shape: "<< W2->shape()[0] << "x" << W2->shape()[1] <<std::endl;
    auto flattened = flatten(feature_maps_2);
    auto layer1 = sigmoid(add(mul(W1, flattened, 0, 0), b1));
    auto layer2 = add(mul(W2, layer1, 0, 0), b2);
    auto loss = mse_loss(layer2, label);
    auto graph = ComputationGraph::BuildFromOutput(loss);
    pgraph = &graph; // 设置全局指针
    signal(SIGINT, signal_handler); // 注册信号处理函数
    
    if(argc == 2 && std::string(argv[1]) == "resume"){
        // pgraph->LoadArch("mnist_model_arch_interrupt.txt");
        pgraph->LoadParams("test/mnist_model_params_interrupt.txt");
        std::cout << "Loaded model from interrupt files." << std::endl;
    }  
    // pgraph->LoadParams("mnist_model_params_interrupt.txt");
    // ComputationGraph graph = ComputationGraph::BuildFromOutput(loss);
    // graph.SaveArch("mnist_model_arch.txt");
    // exit(0);
    // 收集所有参数
    // std::vector<VarPtr> params = {kernel_1, kernel_2, W1, b1, W2, b2};
    size_t total_params = 0;
    for (const auto& param : pgraph->parameter_nodes) {
        total_params += param->size();
    }
    std::cout << "Total parameters: " << total_params << std::endl;
    std::cout << "Network built successfully!" << std::endl;
    std::cout << "Feature maps 2 shape: ";
    for (size_t dim : feature_maps_2->shape()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    std::cout << "Input size for FC layer: " << input_size << std::endl;
    
    // 训练循环
    const double learning_rate = 0.001;
    const int num_epochs = 2;  // 减少epoch数用于测试
    const int batch_log_interval = 100;
    
    std::cout << "\nStarting training..." << std::endl;
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << std::endl;
        
        double epoch_loss = 0.0;
        int correct_predictions = 0;
        
        // 只使用前5000个样本进行测试
        size_t train_samples = std::min(size_t(5000), dataset.size());
        
        for (size_t i = 0; i < train_samples; i++) {
            // 设置输入数据
            const auto& image_data = dataset.get_image(i);
            auto label_data = dataset.get_one_hot_label(i);
            x->set_input(image_data);

            label->set_input(label_data);
            
            // 前向传播
            loss->zero_grad_recursive();
            loss->calc();
            
            // 记录损失
            double current_loss = loss->item();
            epoch_loss += current_loss;
            
            // 简单的准确率计算（找到最大输出）
            const auto& predictions = layer2->data();
            int predicted_class = 0;
            double max_prob = predictions[0];
            for (int j = 1; j < n_output; j++) {
                if (predictions[j] > max_prob) {
                    max_prob = predictions[j];
                    predicted_class = j;
                }
            }
            bool result= predicted_class == dataset.get_label(i) ? 1 : 0;
            results.push_back(result);
            if (result) {
                correct_predictions++;
            }
            double accuracy_past50 = 0.0;
            if (results.size() >= 50) {
                int correct_past50 = std::count(results.end() - 50, results.end(), true);
                accuracy_past50 = static_cast<double>(correct_past50) / 50.0 * 100.0;
            } else {
                int correct_so_far = std::count(results.begin(), results.end(), true);
                accuracy_past50 = static_cast<double>(correct_so_far) / results.size() * 100.0;
            }
            double accuracy_past100 = 0.0;
            if (results.size() >= 100) {
                int correct_past100 = std::count(results.end() - 100, results.end(), true);
                accuracy_past100 = static_cast<double>(correct_past100) / 100.0 * 100.0;
            } else {
                int correct_so_far = std::count(results.begin(), results.end(), true);
                accuracy_past100 = static_cast<double>(correct_so_far) / results.size() * 100.0;
            }
            double accuracy_past500 = 0.0;
            if (results.size() >= 500) {
                int correct_past500 = std::count(results.end() - 500, results.end(), true);
                accuracy_past500 = static_cast<double>(correct_past500) / 500.0 * 100.0;
            } else {
                int correct_so_far = std::count(results.begin(), results.end(), true);
                accuracy_past500 = static_cast<double>(correct_so_far) / results.size() * 100.0;
            }
            print_vec(std::cout,predictions);
            // 打印进度
            std::cout<< "i: " << i << ", loss: " << std::fixed << std::setprecision(6) << current_loss << " prev_pred:" << predicted_class << " real:" << dataset.get_label(i) << " accuracy: " << std::fixed << std::setprecision(2) << (double(correct_predictions) / (i + 1) * 100.0) << "%" << " acc_p50:" << std::setprecision(2) << accuracy_past50 << "%" << " acc_p100:" << std::setprecision(2) << accuracy_past100 << "%" << " acc_p500:" << std::setprecision(2) << accuracy_past500 << "%" << "\r";
            std::cout.flush();
            if (i % batch_log_interval == 0) {
                double avg_loss = epoch_loss / (i + 1);
                double accuracy = double(correct_predictions) / (i + 1) * 100.0;
                std::cout << "  Step " << i << "/" << train_samples 
                          << ", Loss: " << std::fixed << std::setprecision(6) << current_loss
                          << ", Avg Loss: " << avg_loss
                          << ", Accuracy: " << std::setprecision(2) << accuracy << "%" 
                          << std::endl;
            }
            
            // 反向传播
           loss->backward();
            
            // 参数更新
            for (auto& param : pgraph->parameter_nodes) {
                param->update(learning_rate);
            }
        }
        
        // Epoch总结
        double avg_epoch_loss = epoch_loss / train_samples;
        double epoch_accuracy = double(correct_predictions) / train_samples * 100.0;
        
        std::cout << "Epoch " << epoch + 1 << " Summary:" << std::endl;
        std::cout << "  Average Loss: " << std::fixed << std::setprecision(6) << avg_epoch_loss << std::endl;
        std::cout << "  Accuracy: " << std::setprecision(2) << epoch_accuracy << "%" << std::endl;
        std::cout << "  Correct: " << correct_predictions << "/" << train_samples << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "Training completed!" << std::endl;
    pgraph->SaveParams("test/mnist_model_params.txt");
    std::cout << "Model parameters saved to mnist_model_params.txt" << std::endl;
    return 0;
}