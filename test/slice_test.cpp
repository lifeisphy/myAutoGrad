#include "../autograd.hpp"
#include "../utils.hpp"
#include <iomanip>
using namespace std;

int main(){
    cout << "开始神经网络训练..." << endl;
    
    // 创建固定的输入和目标权重
    auto W = make_param(vec_r(10 * 10 * 5 * 3, 0.1), {10,10,5,3});
    
    auto ret = parse_slices(":-1, ::-1, 1:-1:2, -1",W->shape());
    auto a = slice_(W, ret,"W_slice");
    for(const auto &t: ret){
        cout << std::get<0>(t) << ":" << std::get<1>(t) << ":" << std::get<2>(t) << endl;
    }
    return 0;
}