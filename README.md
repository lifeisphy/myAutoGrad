# MyAutoGrad - C++ Automatic Differentiation Framework

[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B23)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-latest-brightgreen.svg)](docs/)

A high-performance automatic differentiation framework implemented in C++, supporting the construction and training of deep neural networks. This framework implements a complete backpropagation algorithm and supports automatic differentiation for scalars, vectors, and multi-dimensional tensors.

> **Note**: This project is based on AI-generated code with modifications. Please use with caution.

## ✨ Key Features

### 🧠 Core Functionality

- **Variable Class**: Core data structure supporting automatic differentiation
- **Computation Graph Management**: Automatic construction and management of computation graphs
- **Backpropagation**: Efficient gradient computation and propagation
- **Memory Management**: High-efficiency memory management using smart pointers and DataView

### 🔧 Supported Operations

- **Basic Math Operations**: Addition, subtraction, multiplication, division, power operations, etc.
- **Activation Functions**: ReLU, Sigmoid, Tanh, Leaky ReLU
- **Loss Functions**: Mean Squared Error (MSE), Binary Cross-Entropy (BCE)
- **Tensor Operations**: Convolution, pooling, slicing, concatenation, flattening, etc.
- **Vector Operations**: Support for vector operations and broadcasting

### 🚀 Advanced Features

- **Recurrent Neural Networks**: Support for RNN and LSTM architectures
- **Optimizer**: Adam optimizer
- **Visualization**: Computation graph visualization functionality
- **Parameter Saving/Loading**: Model parameter persistence
- **Python Bindings**: Python interface through cppyy

## 📁 Project Structure

```{text}
├── autograd.hpp        # Main framework header file
├── variable.hpp        # Variable class definition
├── operations.hpp      # Mathematical operations implementation
├── graph.hpp          # Computation graph management
├── optimizer.hpp      # Optimizer implementation
├── dataview.hpp       # DataView class
├── utils.hpp          # Utility functions
├── recurrent.hpp      # Recurrent neural network support
├── run.py             # Build script
├── requirements.txt   # Python dependencies
├── test/              # Tests and examples
│   ├── demo.cpp       # Basic demonstration
│   ├── test.cpp       # Unit tests
│   ├── mnist.cpp      # MNIST handwritten digit recognition
│   └── ...            # Other test files
└── docs/              # Documentation directory
    ├── overview.md    # Overview documentation
    ├── api/           # API documentation
    ├── examples/      # Example code
    └── architecture.md # Architecture design
```

## 🚀 Quick Start

### Build Requirements

- C++23 compatible compiler (recommended GCC 13+ or Clang 16+)
- Python 3.8+ (for running [run.py](./run.py), optional for Python bindings)

### Install Dependencies

You may work with python and use c++ part as a standalone library. You can also use the provided `run.py` script to build and run the C++ programs.
```bash
# Install Python dependencies (optional)
pip install -r requirements.txt
```

### Build and Run

Currently only mnist project is supported in run.py. You can also directly compile and run the C++ files in the `test/` directory.
```bash
python run.py [download|compile|train|validate] <project_name>
# Example: Download MNIST dataset
python run.py download mnist
```

### Simple Example

```cpp
#include "autograd.hpp"

int main() {
    // Create variables
    auto x = make_param(2.0);
    auto w = make_param(3.0);
    auto b = make_param(1.0);
    
    // Build computation graph: y = w * x + b
    auto y = add(mul(w, x), b);
    
    // Forward computation
    y->calc();
    std::cout << "y = " << y->item() << std::endl;  // Output: y = 7
    
    // Backpropagation
    y->backward();
    std::cout << "dw = " << w->grad_item() << std::endl;  // Output: dw = 2
    std::cout << "dx = " << x->grad_item() << std::endl;  // Output: dx = 3
    
    return 0;
}
```

## 📚 Documentation

- [📖 Overview](docs/overview.md) - Framework introduction and quick start
- [🔧 API Reference](docs/api/README.md) - Detailed API documentation
- [💡 Examples](docs/examples/README.md) - Practical application examples
- [🏗️ Architecture Design](docs/architecture.md) - Framework internal design documentation

## 🎯 Usage Examples

### Linear Regression

```cpp
// Create variables
auto x = make_input(0.0);
auto w = make_param(0.1);
auto b = make_param(0.1);
auto target = make_input(0.0);

// Build model
auto y_pred = add(mul(w, x), b);
auto loss = mse_loss(y_pred, target);

// Training loop
for (int epoch = 0; epoch < 100; ++epoch) {
    x->set_input(training_data[epoch]);
    target->set_input(labels[epoch]);
    
    loss->zero_grad_recursive();
    loss->calc();
    loss->backward();
    
    w->update(learning_rate);
    b->update(learning_rate);
}
```

### Computation Graph

```cpp
make_input(x, init_zeros(input_size), {input_size});
make_input(label, init_zeros(output_size), {output_size});
make_param(b, init_weights(output_size), {output_size});
make_param(W, init_weights(input_size * output_size) , {input_size, output_size});
output = add(mul(W, x, 0, 0), b);
loss = mse_loss(output, label,"loss");
// MSE_LOSS(loss,output,label); the corresponding macro, same functionality but helps to set the name of new node with the variable name 'loss'
auto graph = ComputationGraph::BuildFromOutput(loss); // graph construction from the final output node
// graph->visualize("computation_graph.dot"); // visualize the computation graph to a dot file
graph.toposort(); // returns a list of topological sort on the computation graph
graph.input_nodes; // list of all input nodes
graph.param_nodes; // list of all parameter nodes
graph.output_nodes; // the output nodes
// ...
graph.Visualize("computation_graph.dot"); // visualize the computation graph to a dot file

graph.LoadParams("1.txt"); // load model parameters from file
graph.fit([](ComputationGraph* pgraph){/* load training data for each sample here...*/}, 100, 1000, 0.01) // train the model with 100 epochs, 1000 samples, learning rate 0.01
graph.SaveParams("1.txt"); // save model parameters to file
```


### Neural Network

```cpp
// Multi-layer perceptron
auto W1 = make_param(vec_r(input_size * hidden_size), {hidden_size, input_size});
auto b1 = make_param(vec_r(hidden_size), {hidden_size});
auto W2 = make_param(vec_r(hidden_size * output_size), {output_size, hidden_size});
auto b2 = make_param(vec_r(output_size), {output_size});

// Forward propagation
auto z1 = add(mul(W1, x, 0, 0), b1);
auto a1 = relu(z1);
auto z2 = add(mul(W2, a1, 0, 0), b2);
auto output = z2;
```

### Convolutional Neural Network

```cpp
// Convolution layer
auto conv_weights = make_param(vec_r(3 * 3 * 32), {3, 3, 32});
auto conv_out = conv2d(input, conv_weights);
auto relu_out = relu(conv_out);
auto pool_out = MaxPooling(relu_out, 2);
```

### Recurrent Neural Network

```cpp
// LSTM
auto lstm = RecurrentOperation(lstm_(long_term_size, short_term_size), hidden_state, input); // Create LSTM operation
lstm.expand(seq_length,false, [hidden_dim](VarPtr v){
    return Linear(hidden_dim, 10,false)(v);
}); // expand the LSTM for the given sequence length, false means only 1 output, the output_transform function defines the operation to perform on hidden layer to get each output.
auto outputs = lstm.outputs;
auto graph = ComputationGraph::BuildFromOutput(lstm.outputs); // Build computation graph from the outputs
```

## 🔄 Recent Updates

Recent major updates include:

- **🔄 RNN Support**: Added recurrent neural networks and LSTM support
- **📊 Visualization**: New computation graph visualization functionality
- **⚡ Adam Optimizer**: Implemented Adam optimization algorithm
- **💾 Parameter Saving/Loading**: Support for model parameter persistence
- **🐍 Python Bindings**: Python interface through cppyy
- **🔧 Advanced Tensor Operations**: Convolution, pooling, slicing, and other operations

## 🤝 Contributing

Issues and pull requests are welcome to improve this project!

### Development Guide

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use MyAutoGrad in your research or projects, please consider citing this repository:

```bibtex
@software{myautograd,
  title={MyAutoGrad: C++ Automatic Differentiation Framework},
  author={lifeisphy},
  year={2025},
  url={https://github.com/lifeisphy/myAutoGrad}
}
```

## 🙏 Acknowledgments

- Thanks to all contributors for their support
- Inspired by excellent frameworks like PyTorch and TensorFlow
- Special thanks to the open-source community

## 📞 Contact

- 📧 Email: [lifeisphy@gmail.com]
- 🐛 Issues: [GitHub Issues](https://github.com/lifeisphy/myAutoGrad/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/lifeisphy/myAutoGrad/discussions)

---

⭐ If this project is helpful to you, please give us a Star!
