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

std::string outfile_interrupt = "out/mnist_model_params_interrupt.txt";
std::string savefilename = "out/mnist_model_params.txt";
std::string wrong_preds_file = "out/mnist_wrong_preds_interrupt.txt";
bool loadparams = false;
bool validate_ = false;
bool train_ = false;

struct wrong_pred {
    int idx;
    int wrong_number;
    int real_number;
    std::vector<double> pred_probs;
};
std::vector<wrong_pred> wrong_preds;
ComputationGraph* pgraph = nullptr; // Global pointer to the computation graph
void signal_handler(int signal){
    if(signal == SIGINT){
        if(validate_){
            std::cout << "\nInterrupted by user." << std::endl ;
            std::cout<<"Saving wrong predictions to " << wrong_preds_file << std::endl;
            std::ofstream wrong_file(wrong_preds_file);
            wrong_file << "Index,Predicted,Real,Probabilities\n";
            for(size_t i=0; i< wrong_preds.size(); i++){
                wrong_file << wrong_preds[i].idx << "," << wrong_preds[i].wrong_number << "," << wrong_preds[i].real_number << ",";
                print_vec(wrong_file, wrong_preds[i].pred_probs);
                wrong_file << "\n";
            }
        }else if(train_){
            std::cout << "\nInterrupted by user." << std::endl <<"saving to " << outfile_interrupt << std::endl;
            pgraph->SaveParams(outfile_interrupt);
        }
        exit(0);
    }
    if(signal == SIGQUIT){
        std::cout << "\nInterrupted by user." << std::endl ;
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
// 网络参数
const int n = 28;
const int n_input = n * n;
const int n_output = 10;
const int n_kernel = 32;
const int n_kernel_2 = 48;

size_t input_size;
std::vector<bool> results;
MNISTDataset dataset;

void build_network(){
     // 创建网络变量
    // auto x = make_input(std::vector<double>(n_input, 0.0), {n, n});
    // auto label = make_input(std::vector<double>(n_output, 0.0), {n_output});
    MAKE_INPUT(x, std::vector<double>(n_input, 0.0), {n,n});
    MAKE_INPUT(label, std::vector<double>(n_output, 0.0), {n_output});
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
    MAKE_PARAM(kernel_1, init_weights(3 * 3 * n_kernel), {3, 3, n_kernel});
    MAKE_PARAM(kernel_2, init_weights(3 * 3 * n_kernel_2 * n_kernel), {3, 3, n_kernel, n_kernel_2});
    // auto kernel_1 = make_param(init_weights(3 * 3 * n_kernel), {3, 3, n_kernel});
    // auto kernel_2 = make_param(init_weights(3 * 3 * n_kernel_2 * n_kernel), {3, 3, n_kernel, n_kernel_2});
    
    // 第一个卷积层
    std::cout << "Building first conv layer..." << std::endl;
    std::vector<VarPtr> output_1;
    for (int i = 0; i < n_kernel; i++) {
        SLICE(kernel_slice, kernel_1, {-1, -1, i});
        CONV2D(conv_out, x, kernel_slice);
        RELU(relu_out, conv_out);
        MAXPOOLING(pool_out, relu_out, 2);
        // auto kernel_slice = slice(kernel_1, {-1, -1, i});  // 提取第i个卷积核
        // auto conv_out = conv2d(x, kernel_slice);
        // auto relu_out = relu(conv_out);
        // auto pool_out = MaxPooling(relu_out, 2);
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
            SLICE(kernel_slice, kernel_2, {-1, -1, j, i});
            CONV2D(conv_out, slices_1[j], kernel_slice);
            RELU(relu_out, conv_out);
            MAXPOOLING(pool_out, relu_out, 2);
            // conv_results.push_back(pool_out);
            // auto kernel_slice = slice(kernel_2, {-1, -1, j, i});
            // auto conv_out = conv2d(slices_1[j], kernel_slice);
            // auto relu_out = relu(conv_out);
            // auto pool_out = MaxPooling(relu_out, 2);
            conv_results.push_back(pool_out);
        }
        SUM(summed, conv_results);
        // auto summed = sum(conv_results);
        output_2.push_back(summed);
    }
    // auto feature_maps_2 = stack(output_2);
    STACK(feature_maps_2, output_2);
    std::cout<<"Feature maps 2 shape: " << feature_maps_2->shape()[0] << "x" << feature_maps_2->shape()[1] << "x" << feature_maps_2->shape()[2] << std::endl;
    // 全连接层
    std::cout << "Building fully connected layers..." << std::endl;
    input_size = feature_maps_2->size();
    const int mid_size = 128;
    std::cout<<"input size: "<< input_size <<std::endl;
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
    auto graph = new ComputationGraph();
    *graph = ComputationGraph::BuildFromOutput(loss);
    pgraph = graph;
    size_t total_params = 0;
    for (const auto& param : pgraph->parameter_nodes) {
        std::cout<<"Param: " << param->name << " Shape: ";
        print_vec(std::cout, param->shape());
        std::cout << " Size: " << param->size() << std::endl;
        total_params += param->size();
    }
    std::cout << "Total parameters: " << total_params << std::endl;
    std::cout << "Network built successfully!" << std::endl;

}
void train(){
    // 训练循环
    const double learning_rate = 0.001;
    const int num_epochs = 2;  // 减少epoch数用于测试
    const int batch_log_interval = 100;
    const size_t num_samples = dataset.size();
    std::cout << "\nStarting training..." << std::endl;

    double current_loss = 0.0;
    double epoch_loss = 0.0;
    int correct_predictions = 0;
    auto x = pgraph->get_node_by_name("x");
    auto label = pgraph->get_node_by_name("label");
    auto output = pgraph->get_node_by_name("output");
    auto load_data = [&](ComputationGraph* g) {
        // 这里实现数据加载逻辑
        const auto& image_data = dataset.get_image(g->i);
        auto label_data = dataset.get_one_hot_label(g->i);
        x->set_input(image_data);
        label->set_input(label_data);
    };
    auto print_info_before = [&](ComputationGraph* g) {
        std::cout<<"i: " << g->i ;
    };
    auto print_info_after = [&](ComputationGraph* g) {

        std::cout << " loss: " <<std::fixed << std::setprecision(6) << g->output_nodes[0]->item();
        epoch_loss += current_loss;
        // 简单的准确率计算（找到最大输出）
        const auto& predictions = output->data();
        int predicted_class = 0;
        double max_prob = predictions[0];
        for (int j = 1; j < n_output; j++) {
            if (predictions[j] > max_prob) {
                max_prob = predictions[j];
                predicted_class = j;
            }
        }
        int real = dataset.get_label(g->i);
        bool result= predicted_class == real;
        results.push_back(result);
        if(result)
            correct_predictions +=1;
        auto get_acc = [](int pastN, const std::vector<bool>& results) {
            if (results.size() >= pastN) {
                int correct_count = std::count(results.end() - pastN, results.end(), true);
                return static_cast<double>(correct_count) / pastN * 100.0;
            } else {
                int correct_so_far = std::count(results.begin(), results.end(), true);
                return static_cast<double>(correct_so_far) / results.size() * 100.0;
            }
        };
        double acc_p50 = get_acc(50, results);
        double acc_p100 = get_acc(100, results);
        double acc_p500 = get_acc(500, results);
        std::cout<<std::setprecision(2);
        print_vec(std::cout,predictions);
        std::cout<<"pred: "<< predicted_class << " real: " << real;

        std::cout<< " acc: " << (double(correct_predictions) / (g->i + 1) * 100.0);
        std::cout << "% acc_p50:" << acc_p50;
        std::cout << "% acc_p100:" << acc_p100;
        std::cout << "% acc_p500:" << acc_p500;
        std::cout << "% \r";
        std::cout.flush();
    };
    pgraph->fit(
    load_data, num_epochs, num_samples, learning_rate,
    print_info_before, print_info_after);
    
    std::cout << "Training completed!" << std::endl;
    pgraph->SaveParams(savefilename);
    std::cout << "Model parameters saved to " << savefilename << std::endl;
}

int main(int argc, char* argv[]) {

    std::cout << "=== MNIST CNN Training in C++ ===" << std::endl;

    // 加载数据
    auto res = dataset.load_csv("testcases/mnist/Kaggle-Digit-Recognizer-master/train.csv");
    // auto res = dataset.load_csv("testcases/digit-recognizer/train.csv");
    if (!res) {
        return -1;
    }

    build_network();

    

    if (argc == 1){
        std::cout << "Usage: " << argv[0] << " load <model_params_file> validate|train" << std::endl;
    }
    for(int i=1; i < argc; ){
        auto cmd = std::string(argv[i]);
        if(cmd == "load"){
            auto arg = std::string(argv[i+1]);
            pgraph->LoadParams(arg);
            loadparams = true;
            std::cout << "Loaded model from " << arg << std::endl;
            i += 2;
        }else if(cmd == "validate"){
            validate_ = true;
            i+= 1;
        }else if(cmd == "train"){
            train_ = true;
            i += 1;
        }else {
            std::cout << "Unknown command: " << cmd << std::endl;
            exit(1);
        }
    }
    signal(SIGINT, signal_handler); // 注册信号处理函数
    signal(SIGQUIT, signal_handler);
    if(train_){
        std::cout << "Starting training..." << std::endl;
        train();
    }else if(validate_){

        std::cout << "Starting validation..." << std::endl;
        if(!loadparams){
            std::cout << "Please load model parameters before validation." << std::endl;
            return -1;
        }
        auto x = pgraph->get_node_by_name("x");
        auto label = pgraph->get_node_by_name("label");
        auto output = pgraph->get_node_by_name("output");
        int correct_predictions = 0;
        for(int i=0; i< dataset.size(); i++){
            // pgraph->i = i;
            output->zero_grad_recursive();
            x->set_input( dataset.get_image(i));
            output->calc(); 
            auto predictions = output->data();
            std::vector<double> probs;
            for(int i=0; i< predictions.size(); i++){
                probs.push_back(predictions[i]);
            }
            int predicted_class = 0;
            double max_prob = predictions[0];
            for (int j = 1; j < n_output; j++) {
                if (predictions[j] > max_prob) {
                    max_prob = predictions[j];
                    predicted_class = j;
                }
            }
            int real = dataset.get_label(i);
            bool result= predicted_class == real;
            results.push_back(result);
            if(result)
                correct_predictions +=1;
            else{
                wrong_pred wp;
                wp.idx = i;
                wp.wrong_number = predicted_class;
                wp.real_number = real;
                wp.pred_probs = probs;
                wrong_preds.push_back(wp);
            }
            std::cout<<"i: " << i ;
            std::cout<<"pred: "<< predicted_class << " real: " << real << (result? "✔️":"❌");
            std::cout<<std::setprecision(6);
            std::cout<< " acc: " << (double(correct_predictions) / (i + 1) * 100.0) << "% ";
            std::cout<<std::setprecision(2);
            print_vec(std::cout,predictions);
            std::cout<<"\n";
            std::cout.flush();
        }
        std::cout<<"\nValidation completed!" << std::endl;
        std::cout << "Final accuracy: " << (double(correct_predictions) / dataset.size() * 100.0) << "%" << std::endl;
    }
    

}