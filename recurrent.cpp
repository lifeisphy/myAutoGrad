#include "autograd.hpp"
#include "operations.hpp"
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
    // function signature: fn(hidden_state, input, make_params, params)
    // if make_params is true, fn will create parameters and append to params vector
    // otherwise, fn will use the existing params in the vector
    recurrent_op fn;
    RecurrentOperation(recurrent_op fn_, VarPtr hidden_input, VarPtr input)
        : hidden_input(hidden_input), input(input), fn(fn_) {
        hidden_output = fn(hidden_input, input, true, params);

        hidden.push_back(hidden_input);
        hidden.push_back(hidden_output);
        inputs.push_back(input);
    }

    // Expand the recurrent operation for given times
    // @param times: resulting number of inputs (time steps)
    // len(inputs)== times, len(hidden) == times + 1, 
    // len(outputs) == times if multiple_outputs==true else 1
    void expand(int times, bool multiple_outputs = false, std::function<VarPtr(VarPtr)> transform = nullptr){
        VarPtr h_prev = hidden_output;
        size_t input_size = input->size();
        std::vector<size_t> input_shape = input->shape();
        for(int t=0; t < times - 1; t++){
            if(multiple_outputs){
                // y_{t} = h_{t+1}
                VarPtr out;
                if(transform) {
                    out = transform(h_prev);
                }else {
                    out = h_prev;
                }
                outputs.push_back(out);
            }
            auto in = make_input(zero_vec(input_size), input_shape, input->name + std::to_string(t+1));
            VarPtr h_new = fn(h_prev, in, false, params); // use existing params
            inputs.push_back(in);
            hidden.push_back(h_new);
            h_prev = h_new;
        }
        // y_{n-1}=h_n
        VarPtr out;
        if(transform) {
            out = transform(h_prev);
        }else {
            out = h_prev;
        }
        outputs.push_back(out);

    }
};
recurrent_op linear_( bool use_bias=true){
    return [=](VarPtr hidden_state, VarPtr input, bool make_params, std::vector<VarPtr>& params) -> VarPtr {
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
        size_t hidden_dim = hidden_state->size(); // hidden_state contains both h(long) and c(short)
        assert(hidden_dim == long_term_size + short_term_size);
        size_t input_dim = input->size();
        auto h = slice_(hidden_state, {0}, {long_term_size}, "lstm_h_prev");
        auto c = slice_(hidden_state, {long_term_size}, {hidden_dim}, "lstm_c_prev");
        size_t combined_dim = short_term_size + input_dim; // the dimension of short term state + input
        auto combined = stack({c, input}, "lstm_combined");
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
        return concat({ h_new, c_new}, "lstm_hidden_output");
    };
}
