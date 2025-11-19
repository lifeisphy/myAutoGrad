#pragma once
#include "../autograd.hpp"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <regex>
#include <filesystem>

class IMDBDataset {
private:
    std::vector<std::vector<int>> sequences_;  // 词索引序列
    std::vector<int> labels_;                  // 标签 (0=负面, 1=正面)
    std::unordered_map<std::string, int> word2idx_;
    std::vector<std::string> idx2word_;
    int vocab_size_;
    int max_sequence_length_;
    
public:
    IMDBDataset(int max_vocab = 10000, int max_seq_len = 500) 
        : vocab_size_(max_vocab), max_sequence_length_(max_seq_len) {
        // 添加特殊标记
        word2idx_["<PAD>"] = 0;
        word2idx_["<UNK>"] = 1;
        word2idx_["<START>"] = 2;
        word2idx_["<END>"] = 3;
        idx2word_ = {"<PAD>", "<UNK>", "<START>", "<END>"};
    }
    
    // 文本预处理
    std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::string cleaned = text;
        
        // 转换为小写
        std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);
        
        // 移除HTML标签
        std::regex html_tags("<[^>]*>");
        cleaned = std::regex_replace(cleaned, html_tags, " ");
        
        // 移除标点符号，保留字母和数字
        std::regex punct("[^a-zA-Z0-9\\s]");
        cleaned = std::regex_replace(cleaned, punct, " ");
        
        // 分词
        std::istringstream iss(cleaned);
        std::string word;
        while (iss >> word) {
            if (!word.empty()) {
                tokens.push_back(word);
            }
        }
        
        return tokens;
    }
    
    // 构建词汇表
    void build_vocabulary(const std::vector<std::string>& texts) {
        std::unordered_map<std::string, int> word_freq;
        
        // 统计词频
        for (const auto& text : texts) {
            auto tokens = tokenize(text);
            for (const auto& token : tokens) {
                word_freq[token]++;
            }
        }
        
        // 按频率排序
        std::vector<std::pair<std::string, int>> freq_pairs(word_freq.begin(), word_freq.end());
        std::sort(freq_pairs.begin(), freq_pairs.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        // save to out/vocab.txt
        std::cout<<"Writing vocabulary to out/vocab_freq.txt" << std::endl;
        std::ofstream vocab_file("out/vocab_freq.txt");
        for (const auto& pair : freq_pairs) {
            vocab_file << pair.first << " " << pair.second << std::endl;
        }
        vocab_file.close();
        // 构建词汇表（保留最高频的词）
        int idx = idx2word_.size();
        for (const auto& pair : freq_pairs) {
            if (idx >= vocab_size_) break;
            word2idx_[pair.first] = idx;
            idx2word_.push_back(pair.first);
            idx++;
        }
        
        std::cout << "Vocabulary built: " << idx2word_.size() << " words" << std::endl;
    }
    
    // 文本转换为序列
    std::vector<int> text_to_sequence(const std::string& text) {
        auto tokens = tokenize(text);
        std::vector<int> sequence;
        sequence.push_back(word2idx_["<START>"]);
        
        for (const auto& token : tokens) {
            auto it = word2idx_.find(token);
            if (it != word2idx_.end()) {
                sequence.push_back(it->second);
            } else {
                sequence.push_back(word2idx_["<UNK>"]);
            }
            
            if (sequence.size() >= max_sequence_length_ - 1) break;
        }
        
        sequence.push_back(word2idx_["<END>"]);
        
        // 填充到固定长度
        while (sequence.size() < max_sequence_length_) {
            sequence.push_back(word2idx_["<PAD>"]);
        }
        
        return sequence;
    }
    
    // 加载IMDB数据集
    bool load_data(const std::string& data_path) {
        std::vector<std::string> all_texts;
        std::vector<int> all_labels;
        
        // 加载正面评论
        std::string pos_path = data_path + "/train/pos/";
        std::string neg_path = data_path + "/train/neg/";
        
        // 这里需要你根据实际的文件结构来调整
        // IMDB数据集通常是每个评论一个txt文件
        std::fstream file;
        // list directory and read files
        std::cout<<"Reading files from: " << pos_path << std::endl;
        int max = 10;
        for(auto file: std::filesystem::directory_iterator(pos_path)) {
            if (max-- == 0) break;
            // std::cout<< "Reading file: " << file.path() << std::endl;
            std::ifstream ifs(file.path());
            if (!ifs.is_open()) continue;
            std::stringstream buffer;
            buffer << ifs.rdbuf();
            all_texts.push_back(buffer.str());
            all_labels.push_back(1);  // 正面评论
            ifs.close();
        }
        // file.open(pos_path);
        // for (int i = 0; i < 12500; i++) {  // 假设有12500个正面评论
        //     std::string filename = pos_path + std::to_string(i) + "_*.txt";
        //     // 实际实现中，你需要遍历目录中的所有文件
        //     // 这里只是示例结构
        // }
        
        std::cout << "Loaded " << all_texts.size() << " reviews" << std::endl;
        
        // 构建词汇表
        build_vocabulary(all_texts);
        
        // 转换为序列
        for (size_t i = 0; i < all_texts.size(); i++) {
            sequences_.push_back(text_to_sequence(all_texts[i]));
            labels_.push_back(all_labels[i]);
        }
        
        return true;
    }
    
    // 简化版：从CSV加载（更容易处理）
    bool load_csv(const std::string& csv_file) {
        std::ifstream file(csv_file);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << csv_file << std::endl;
            return false;
        }
        
        std::vector<std::string> all_texts;
        std::vector<int> all_labels;
        
        std::string line;
        std::getline(file, line); // 跳过header
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string label_str, review;
            
            std::getline(ss, label_str, ',');
            std::getline(ss, review);
            
            // 移除引号
            if (!review.empty() && review.front() == '"') {
                review = review.substr(1);
            }
            if (!review.empty() && review.back() == '"') {
                review = review.substr(0, review.length() - 1);
            }
            
            all_texts.push_back(review);
            all_labels.push_back(std::stoi(label_str));
        }
        
        file.close();
        
        std::cout << "Loaded " << all_texts.size() << " reviews from CSV" << std::endl;
        
        // 构建词汇表
        build_vocabulary(all_texts);
        
        // 转换为序列
        for (size_t i = 0; i < all_texts.size(); i++) {
            sequences_.push_back(text_to_sequence(all_texts[i]));
            labels_.push_back(all_labels[i]);
        }
        
        return true;
    }
    
    size_t size() const { return sequences_.size(); }
    const std::vector<int>& get_sequence(size_t idx) const { return sequences_[idx]; }
    int get_label(size_t idx) const { return labels_[idx]; }
    std::string get_word(int idx) const { 
        if (idx >= 0 && idx < idx2word_.size()) {
            return idx2word_[idx];
        }
        return "<UNK>";
    }
    int get_word_index(const std::string& word) const {
        auto it = word2idx_.find(word);
        if (it != word2idx_.end()) {
            return it->second;
        }
        return word2idx_.at("<UNK>");
    }
    int get_vocab_size() const { return idx2word_.size(); }
    int get_sequence_length() const { return max_sequence_length_; }
};


class SentimentLSTM {
    public:
    RecurrentOperation* lstm_op_;
    VarPtr label;
    VarPtr loss;
    VarPtr real_output_node;
    ComputationGraph graph;
    std::vector<VarPtr> inputs, hiddens, outputs;
    size_t vocab_size_;
    SentimentLSTM(size_t vocab_size,size_t embedding_dim, size_t max_expand, size_t hidden_dim, size_t output_dim){
        size_t short_term_size = hidden_dim / 2;
        size_t long_term_size = hidden_dim - short_term_size;
        // size_t max_expand = 100; // 最大展开步数
        auto hidden = make_input(vec_r(hidden_dim,0.0), {hidden_dim}, "lstm_hidden_0");
        auto input = make_input(vec_r(vocab_size,0.0), {vocab_size}, "lstm_input_0");
        auto label = make_input(vec_r(1,0.0), {1}, "label_input");
        Linear classifier = Linear(hidden_dim, output_dim, true);

        Linear embedding = Linear(vocab_size, embedding_dim, false);
        lstm_op_ = new RecurrentOperation(lstm_(long_term_size,short_term_size), hidden, input);
        lstm_op_->expand(max_expand, false, 
        [hidden_dim, vocab_size, &classifier, &label](VarPtr v){
            auto h_final = slice_indices(v, {idx_range(0, hidden_dim, 1)}, "final_hidden");
            auto logits = classifier(h_final);
            auto output = sigmoid(logits, "sentiment_output");
            return output;
            // return mse_loss(output, label, "sentiment_loss");
        }, [&embedding](VarPtr in_raw){
            return embedding(in_raw);
        });
        vocab_size_ = vocab_size;
        this->label = label;
        inputs = lstm_op_->inputs;
        hiddens = lstm_op_->hidden;
        outputs = lstm_op_->outputs;
        graph = ComputationGraph::BuildFromOutput(lstm_op_->outputs[max_expand-1]);

    }
    double forward(const std::vector<int>& sequence){
        // real output node and loss changes according to sequence length
        this->real_output_node = outputs[sequence.size()-1];
        for(int i=0; i < sequence.size(); i++){
            std::vector<double> one_hot(vocab_size_, 0.0);
            if(i >= 0 && i < vocab_size_){
                one_hot[sequence[i]] = 1.0;
            }

            inputs[i]->set_input(one_hot);
        }
        loss = mse_loss(outputs[sequence.size()-1], label, "sentiment_loss");
        loss->calc();
        std::cout<<"Loss: "<<loss->data()[0]<<std::endl;
        auto output_data = this->real_output_node->data();
        return output_data[0];
    }
    void fit(const std::vector<int>& sequence, int label){
        this->real_output_node = outputs[sequence.size()-1];
        loss = mse_loss(this->real_output_node, this->label, "sentiment_loss");
        std::cout<<"Zero grad..."<<std::endl;
        std::flush(std::cout);
        loss->zero_grad_recursive();
        for(auto& out: outputs){
            out->zero_grad_recursive();
        }
        check();
        for(int i=0; i < sequence.size(); i++){
            std::vector<double> one_hot(vocab_size_, 0.0);
            if(i >= 0 && i < vocab_size_){
                one_hot[sequence[i]] = 1.0;
            }

            inputs[i]->set_input(one_hot);
        }
        this->label->set_input({static_cast<double>(label)});
        std::cout<<"Calculating loss..."<<std::endl;
        std::flush(std::cout);
        loss->calc();
        std::cout<<"Loss: "<<loss->data()[0]<<std::endl;
    }
    void check(){
        for(auto &param : this->graph.parameter_nodes){
            param->print(std::cout,false);
        }
        for(auto& hidden: this->hiddens){
            hidden->print(std::cout,false);
        }
        for(auto &input : this->inputs){
            input->print(std::cout,false);
        }
        for(auto &output : this->outputs){
            output->print(std::cout,false);
        }
    }
    void backward(){
        loss->backward();
    }
    void update(double learning_rate){
        for(auto &param : this->graph.parameter_nodes){
            // param->print(std::cout,false);
            param->update(learning_rate);
        }
    }
};


// class SentimentLSTM {
// private:
//     int vocab_size_;
//     int embedding_dim_;
//     int hidden_size_;
//     int sequence_length_;
    
//     VarPtr embedding_weights_;
//     RecurrentOperation* lstm_op_;
//     Linear classifier_;
    
// public:
//     SentimentLSTM(int vocab_size, int embedding_dim = 128, int hidden_size = 64, int sequence_length = 500)
//         : vocab_size_(vocab_size), embedding_dim_(embedding_dim), 
//           hidden_size_(hidden_size), sequence_length_(sequence_length) {
        
//         // 初始化嵌入层权重
//         // std::random_device rd;
//         // std::mt19937 gen(42);
//         // std::normal_distribution<double> dist(0.0, 0.1);
        
//         std::vector<double> emb_data(vocab_size * embedding_dim);
//         for (auto& val : emb_data) {
//             val = dist(gen);
//         }
        
//         embedding_weights_ = make_param(emb_data, {vocab_size, embedding_dim}, "embedding_weights");
        
//         // 初始化LSTM
//         auto dummy_hidden = make_input(zero_vec(hidden_size * 2), {hidden_size * 2}, "lstm_hidden_init");
//         auto dummy_input = make_input(zero_vec(embedding_dim), {embedding_dim}, "lstm_input_dummy");
        
//         lstm_op_ = new RecurrentOperation(lstm_(hidden_size, hidden_size), dummy_hidden, dummy_input);
        
//         // 初始化分类器
//         classifier_ = Linear(hidden_size, 1, true);  // 二分类
//     }
    
//     VarPtr forward(const std::vector<int>& sequence, VarPtr label = nullptr) {
//         // 词嵌入
//         std::vector<VarPtr> embeddings;
//         for (int word_idx : sequence) {
//             // 创建one-hot向量
//             std::vector<double> one_hot(vocab_size_, 0.0);
//             if (word_idx >= 0 && word_idx < vocab_size_) {
//                 one_hot[word_idx] = 1.0;
//             }
//             auto word_vec = make_input(one_hot, {vocab_size_}, "word_" + std::to_string(word_idx));
//             auto embedded = mul(embedding_weights_, word_vec, 1, 0, "embedded_word");
//             embeddings.push_back(embedded);
//         }
        
//         // LSTM处理序列
//         VarPtr hidden = make_input(zero_vec(hidden_size_ * 2), {hidden_size_ * 2}, "lstm_initial_hidden");
        
//         for (auto& emb : embeddings) {
//             auto lstm_fn = lstm_(hidden_size_, hidden_size_);
//             hidden = lstm_fn(hidden, emb, false, lstm_op_->params);
//         }
        
//         // 提取最后的隐藏状态用于分类
//         auto h_final = slice_indices(hidden, {idx_range(0, hidden_size_, 1)}, "final_hidden");
        
//         // 分类
//         auto logits = classifier_(h_final);
//         auto output = sigmoid(logits, "sentiment_output");
        
//         if (label) {
//             return mse_loss(output, label, "sentiment_loss");
//         }
        
//         return output;
//     }
    
//     std::vector<VarPtr> get_parameters() {
//         std::vector<VarPtr> params = {embedding_weights_};
//         for (auto p : lstm_op_->params) {
//             params.push_back(p);
//         }
//         params.push_back(classifier_.weights);
//         params.push_back(classifier_.bias);
//         return params;
//     }
// };