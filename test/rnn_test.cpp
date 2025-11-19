#include "../autograd.hpp"
// #include "../utils.hpp"
// #include "../recurrent.hpp"
int main(){
    size_t input_dim = 5;
    size_t hidden_dim = 8;
    size_t long_term_size = 4;
    size_t short_term_size = hidden_dim - long_term_size;
    auto hidden = make_input(vec_r(hidden_dim,0.0), {hidden_dim}, "lstm_hidden_0");
    auto input = make_input(vec_r(input_dim,0.0), {input_dim}, "lstm_input_0");
    auto op = RecurrentOperation(lstm_(long_term_size,short_term_size), hidden, input);
    op.expand(4, false, [hidden_dim](VarPtr v){
        return Linear(hidden_dim, 10,false)(v);
    });
    std::cout << "LSTM Recurrent Operation Expanded:" << std::endl;
    for(size_t t=0; t < op.inputs.size(); t++){
        std::cout << "Time step " << t << ":" << std::endl;
        std::cout << "  Input: " << op.inputs[t]->name << ", shape: ";
        print_vec(std::cout, op.inputs[t]->shape()); std::cout << std::endl;
        std::cout << "  Hidden: " << op.hidden[t]->name << ", shape: ";
        print_vec(std::cout, op.hidden[t]->shape()); std::cout << std::endl;
    }
    std::cout << "  Output: " << op.outputs[0]->name << ", shape: ";
    print_vec(std::cout, op.outputs[0]->shape()); std::cout << std::endl;
    auto graph = ComputationGraph::BuildFromOutput(op.outputs[0]);
    graph.print_summary();
    return 0;
}