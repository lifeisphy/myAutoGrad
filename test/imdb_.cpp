#include "imdb.hpp"
#include <csignal>
using namespace std;
SentimentLSTM* pmodel;
std::string filename = "out/imdb_model_params.txt";
void sigint_handler(int sig) {
    cout << "Interrupt signal (" << sig << ") received. Saving to " << filename << " and exiting..." << endl;
    pmodel->graph.SaveParams(filename);
    exit(0);
}

int main(int argc, char* argv[]) {
    signal(SIGINT, sigint_handler);
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
    pmodel = new SentimentLSTM(dataset.get_vocab_size(), 32, dataset.get_sequence_length(), 128, 1);
    if(argc == 2 && std::string(argv[1]) == "load"){
        pmodel->graph.LoadParams("out/imdb_model_params.txt");
        cout<<"Model parameters loaded from out/imdb_model_params.txt"<<endl;
    }
    // 训练参数
    const double learning_rate = 0.01;
    const int epochs = 1;
    std::cout << "Starting training..." << std::endl;
    auto adam = AdamOptimizer(learning_rate);
    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        int correct_predictions = 0;
        for (int i = 0; i < dataset.size(); i++) {
            std::cout << "Epoch " << epoch + 1 << ", Sample " << i + 1 << "/" << dataset.size() << " String seq:";
            for (auto v: dataset.get_sequence(i)){
                std::cout << dataset.get_word(v) << " ";
            }
            cout << "\r";
            std::flush(cout);
            
            // 准备标签
            int label = dataset.get_label(i);
            std::vector<int> seq = dataset.get_sequence(i);

            cout<<"Fitting: ";
            flush(cout);
            pmodel->fit(seq, label);
            cout<<"Updating: ";
            pmodel->update(learning_rate);
            // pmodel->backward(seq.size()-1);
            double loss = pmodel->losses[seq.size()-1]->data()[0];
            epoch_loss += loss;
            cout<<"loss: "<<loss<<endl;
            // 计算准确率
            // auto output = pmodel->forward(dataset.get_sequence(idx));
            // int predicted = (output > 0.5) ? 1 : 0;
            // if (predicted == dataset.get_label(idx)) {
            //     correct_predictions++;
            // }
        }
        
        epoch_loss /= dataset.size();
        
        double accuracy = static_cast<double>(correct_predictions) / dataset.size();
        std::cout << "Epoch " << epoch << " completed. "
                  << "Average Loss: " << epoch_loss
                  << ", Accuracy: " << accuracy * 100 << "%" << std::endl;
    }

    return 0;
}