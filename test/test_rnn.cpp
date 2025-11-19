#include "../autograd.hpp"
#include "imdb.hpp"
using namespace std;
int main(){
    IMDBDataset dataset(10000, 200);  // 10k词汇表，200序列长度
    cout<< "Loading IMDB dataset..." << endl;
    dataset.load_data("testcases/aclImdb");
    cout << "Dataset loaded. Number of samples: " << dataset.size() << endl;
    cout << "Vocabulary size: " << dataset.get_vocab_size() << endl;
    cout << "Max sequence length: " << dataset.get_sequence_length() << endl;
    // 创建模型
    cout<< "Building Sentiment LSTM model..." << endl;
    SentimentLSTM model(10000,32, 3, 64, 1);
    int epoch = 0;
    int i = 0;
    std::cout << "Epoch " << epoch + 1 << ", Sample " << i + 1 << "/" << dataset.size() << " String seq:";
    for (auto v: dataset.get_sequence(i)){
        std::cout << dataset.get_word(v) << " ";
    }
    auto seq = dataset.get_sequence(i);
    seq.resize(3);
    int label = dataset.get_label(i);

    for(int i = 0 ; i < 100; i++){
        // 前向传播
        cout<<"Forwarding: ";
        model.fit(seq, label);
        // 反向传播
        cout<<"Backwarding: ";
        model.backward();
        model.update(0.01);   
        // 计算准确率
        auto output = model.forward(seq);

    }
}