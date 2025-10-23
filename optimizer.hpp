#include <map>
#include <vector>
#include "autograd.hpp"
#pragma once
class AdamOptimizer {
    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    std::map<VarPtr, std::pair<std::vector<double>, std::vector<double>>> moments_; // first: m, second: v
public:
    AdamOptimizer(double learning_rate=0.001, double beta1=0.9, double beta2=0.999, double epsilon=1e-8)
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}
    void set_parameter_nodes(std::vector<VarPtr> params) {
        for (const auto& param : params) {
            if (param->has_grad()) {
                moments_[param] = {std::vector<double>(param->size(), 0.0), std::vector<double>(param->size(), 0.0)};
            }
        }
    }
    void update(VarPtr& var) {
        if (!var->has_grad()) {
            throw std::runtime_error("Variable does not have gradients for Adam update");
        }

        auto it = moments_.find(var);
        if (it == moments_.end()) {
            throw std::runtime_error("Variable not found in Adam optimizer moments");
        }
        auto& [m, v] = it->second;
        for (size_t i = 0; i < var->size(); ++i) {
            m[i] = beta1_ * m[i] + (1 - beta1_) * var->grad()[i];
            v[i] = beta2_ * v[i] + (1 - beta2_) * var->grad()[i] * var->grad()[i];
            double m_hat = m[i] / (1 - beta1_);
            double v_hat = v[i] / (1 - beta2_);
            var->Item(i) -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
};