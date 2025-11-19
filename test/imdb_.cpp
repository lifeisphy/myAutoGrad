#include "imdb.hpp"
using namespace std;
int main() {
    // 加载数据集
    IMDBDataset dataset(10000, 200);  // 10k词汇表，200序列长度
    cout<< "Loading IMDB dataset..." << endl;
    dataset.load_data("testcases/aclImdb");
    // 你可以下载预处理好的CSV格式数据：
    // https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    // if (!dataset.load_csv("data/IMDB_Dataset.csv")) {
    //     return -1;
    // }
    cout << "Dataset loaded. Number of samples: " << dataset.size() << endl;
    cout << "Vocabulary size: " << dataset.get_vocab_size() << endl;
    cout << "Max sequence length: " << dataset.get_sequence_length() << endl;


    // 创建模型
    cout<< "Building Sentiment LSTM model..." << endl;
    //  200
    SentimentLSTM model(dataset.get_vocab_size(), 32, dataset.get_sequence_length(), 128, 1);
    auto graph = ComputationGraph::BuildFromOutput(model.lstm_op_->outputs[dataset.get_sequence_length()-1]);
    // graph.print_summary();
    // exit(0);
    // 训练参数
    const double learning_rate = 0.001;
    const int epochs = 5;
    // const int batch_size = 32;
    
    std::cout << "Starting training..." << std::endl;
    auto adam = AdamOptimizer(learning_rate);
    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        int correct_predictions = 0;
        // int num_batches = dataset.size() / batch_size;

        for (int i = 0; i < dataset.size(); i++) {
            std::cout << "Epoch " << epoch + 1 << ", Sample " << i + 1 << "/" << dataset.size() << " String seq:";
            for (auto v: dataset.get_sequence(i)){
                std::cout << dataset.get_word(v) << " ";
            }
            cout << "\r";
            std::flush(cout);
            int idx =  i;
            if (idx >= dataset.size()) break;
            
            // 准备标签
            int label = dataset.get_label(idx);
            // std::vector<double> seq = dataset.get_sequence(idx);
            // 前向传播
            cout<<"Forwarding: ";
            flush(cout);
            model.fit(dataset.get_sequence(idx), label);
            
            // 反向传播
            cout<<"Backwarding: ";
            flush(cout);
            model.backward();
            
            epoch_loss += model.real_output_node->grad()[0];
            
            // 计算准确率
            auto output = model.forward(dataset.get_sequence(idx));
            int predicted = (output > 0.5) ? 1 : 0;
            if (predicted == dataset.get_label(idx)) {
                correct_predictions++;
            }
        }
        
        epoch_loss /= dataset.size();
        
        double accuracy = static_cast<double>(correct_predictions) / dataset.size();
        std::cout << "Epoch " << epoch << " completed. "
                  << "Average Loss: " << epoch_loss
                  << ", Accuracy: " << accuracy * 100 << "%" << std::endl;
    }

    return 0;
}