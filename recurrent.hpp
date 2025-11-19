#pragma once
#include "autograd.hpp"
using recurrent_op = std::function<VarPtr(VarPtr hidden_state, VarPtr input, bool make_params, std::vector<VarPtr>& params)>;
class RecurrentOperation {
    public:
    VarPtr hidden_input;
    VarPtr hidden_output;
    VarPtr input;
    std::vector<VarPtr> params;
    
    // useful only after expand is called
    std::vector<VarPtr> hidden;
    std::vector<VarPtr> inputs;
    std::vector<VarPtr> outputs;
    std::function<VarPtr(VarPtr)> output_transform_, input_transform_;
    // function signature: fn(hidden_state, input, make_params, params)
    // if make_params is true, fn will create parameters and append to params vector
    // otherwise, fn will use the existing params in the vector
    recurrent_op fn;
    RecurrentOperation(recurrent_op fn_, VarPtr hidden_input, VarPtr input)
        : hidden_input(hidden_input), input(input), fn(fn_) {
        
    }

    // Expand the recurrent operation for given times
    // @param times: resulting number of inputs (time steps)
    // len(inputs)== times, len(hidden) == times + 1, 
    // len(outputs) == times if multiple_outputs==true else 1
    void expand(int times, bool multiple_outputs = false, std::function<VarPtr(VarPtr)> output_transform = nullptr,
    std::function<VarPtr(VarPtr)> input_transform = nullptr) {
        output_transform_ = output_transform;
        input_transform_ = input_transform;
        
        // hidden_output = fn(hidden_input, input, true, params);
        // hidden.push_back(hidden_input);
        // hidden.push_back(hidden_output);
        // inputs.push_back(input);

        VarPtr h = hidden_input;
        VarPtr hnext;
        for(int t=0; t < times; t++){
            VarPtr raw_in;
            if(t == 0){
                raw_in = input;
            }else {
                raw_in = make_input(zero_vec(input->size()), input->shape(), input->name + std::to_string(t+1));
            }
            VarPtr in = input_transform_ ? input_transform_(raw_in) : raw_in;

            if(t == 0){
                hnext = fn(h, in, true, params); // create params at first step
            }else {
                hnext = fn(h, in, false, params); // use existing params
            }
            VarPtr out = output_transform_ ? output_transform_(hnext) : hnext;
            outputs.push_back(out);
            inputs.push_back(raw_in);
            hidden.push_back(h);
            h = hnext;
        }
        // y_{n-1}=h_n
        VarPtr out = output_transform_ ? output_transform_(hnext) : hnext;
        outputs.push_back(out);
    }
};
recurrent_op linear_( bool use_bias=true){
    return [=](VarPtr hidden_state, VarPtr input, bool make_params, std::vector<VarPtr>& params) -> VarPtr {
        hidden_state->require_all_gradients_ = false;
        auto in = stack({hidden_state, input}, "recurrent_combined");
        
        if(make_params){
            size_t input_dim = hidden_state->size() + input->size();
            size_t output_dim = hidden_state->size();
            auto layer = Linear(input_dim, output_dim, use_bias);
            params.push_back(layer.weights);
            if(use_bias){
                params.push_back(layer.bias);
            }
            return layer(in);
        }else{
            VarPtr weights, bias;
            weights = params[0];
            if(use_bias){
                bias = params[1];
            }
            auto wx =  mul(weights, in, 1,0);
            if(use_bias){
                return add(wx, bias);
            }
            return wx;
        }
        
    };
}
recurrent_op lstm_(size_t long_term_size, size_t short_term_size){
    return [=](VarPtr hidden_state, VarPtr input, bool make_params, std::vector<VarPtr>& params) -> VarPtr {
        hidden_state->require_all_gradients_ = false;
        size_t hidden_dim = hidden_state->size(); // hidden_state contains both h(long) and c(short)
        assert(hidden_dim == long_term_size + short_term_size);
        assert(long_term_size == short_term_size); // for simplicity, we require long and short term sizes to be equal
        size_t input_dim = input->size();
        auto h = slice_indices(hidden_state, {idx_range(0, long_term_size, 1)}, "lstm_h_prev");
        auto c = slice_indices(hidden_state, {idx_range(long_term_size, hidden_dim, 1)}, "lstm_c_prev");
        size_t combined_dim = short_term_size + input_dim; // the dimension of short term state + input
        auto combined = concat(c, input, "lstm_combined");
        Linear lin1, lin2, lin3, lin4;
        if(make_params){
            // 创建参数
            lin1 = Linear(combined_dim, long_term_size, true);
            lin2 = Linear(combined_dim, long_term_size, true);
            lin3 = Linear(combined_dim, long_term_size, true);
            lin4 = Linear(combined_dim, short_term_size, true);
            for(auto p: {lin1.weights, lin1.bias, lin2.weights, lin2.bias,
                         lin3.weights, lin3.bias, lin4.weights, lin4.bias}){
                params.push_back(p);
            }
            
        }else {
            lin1 = Linear(params[0], params[1]);
            lin2 = Linear(params[2], params[3]);
            lin3 = Linear(params[4], params[5]);
            lin4 = Linear(params[6], params[7]);
        }
        auto f_t = sigmoid(lin1(combined), "lstm_f_t");
        auto i_t = sigmoid(lin2(combined), "lstm_i_t");
        auto g_t = tanh_activation(lin3(combined), "lstm_g_t");
        auto o_t = sigmoid(lin4(combined), "lstm_o_t");
        auto h_new = h * f_t + i_t * g_t;
        auto c_new = o_t * tanh_activation(h_new, "lstm_c_new");
        return concat(h_new,c_new, "lstm_hidden_output");
    };
}
