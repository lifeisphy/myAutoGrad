# Quick Start Guide: CNN for MNIST

This guide will help you get started with the CNN implementation for MNIST digit classification.

## Prerequisites

- C++ compiler with C++23 support (g++ 11+ recommended)
- Make (for building)
- Python 3.6+ (optional, for Python examples)

## Build and Run

### 1. Quick Test

Run the CNN demonstration to see the architecture in action:

```bash
make cnn
```

This will:
- Compile the CNN demo program
- Run training for 3 epochs on synthetic data
- Show loss decreasing over time
- Display a summary of the architecture

**Expected output:**
```
================================================================
CNN Architecture for MNIST Dataset - Demonstration
================================================================

1. Model Architecture:
   [Detailed architecture diagram]

2. Parameter Count:
   Total parameters: 53,866

3. Initializing Model...
   ✓ All parameters initialized

4. Building Computation Graph...
   ✓ All layers built successfully

5. Training Demonstration:
   Epoch 1, Sample 1 (digit=0), Loss: 2.3070
   Epoch 1, Sample 5 (digit=4), Loss: 2.3250
   Epoch 1 - Average Loss: 2.3023
   ...
   Epoch 3 - Average Loss: 2.2857

6. Summary:
   ✓ CNN architecture successfully built and tested
   ✓ Forward propagation works correctly
   ✓ Backward propagation computes gradients
   ✓ Parameters update via gradient descent
```

### 2. Build All Programs

```bash
make all
```

This builds:
- `autograd_demo` - Basic autograd demonstrations
- `autograd_test` - Test suite
- `test/mnist_cnn_demo` - CNN demonstration

### 3. Run Individual Programs

```bash
# Run basic autograd demo
./autograd_demo

# Run tests
./autograd_test

# Run CNN demo
./test/mnist_cnn_demo
```

## Understanding the CNN Architecture

### Architecture Overview

```
Input: 28×28×1 grayscale image (784 pixels)
    ↓
[Conv2D] 1→8 channels, 3×3 kernel
    → 26×26×8 = 5,408 values
    ↓
[ReLU] Activation
    ↓
[MaxPool2D] 2×2 window, stride 2
    → 13×13×8 = 1,352 values
    ↓
[Conv2D] 8→16 channels, 3×3 kernel
    → 11×11×16 = 1,936 values
    ↓
[ReLU] Activation
    ↓
[MaxPool2D] 2×2 window, stride 2
    → 5×5×16 = 400 values
    ↓
[Flatten] → 400 features
    ↓
[Dense] 400→128
    ↓
[ReLU] Activation
    ↓
[Dense] 128→10
    ↓
[Softmax + Cross Entropy] → Loss
```

### Key Features

1. **Automatic Differentiation**: All operations support backpropagation
2. **Shape Preservation**: Tensors maintain their shapes through layers
3. **Efficient**: Uses forward/backward functions for computation
4. **Modular**: Easy to modify architecture

## Using the CNN in Your Code

### Basic Example

```cpp
#include "autograd.hpp"
#include <random>

// 1. Initialize parameters
auto conv1_kernel = make_param(random_data(8*1*3*3), {8, 1, 3, 3});
auto conv1_bias = make_param(zeros(8), {8});
auto fc_weight = make_param(random_data(128*400), {128, 400});
auto fc_bias = make_param(zeros(128), {128});
auto output_weight = make_param(random_data(10*128), {10, 128});
auto output_bias = make_param(zeros(10), {10});

// 2. Create input placeholders
auto input = make_input(zeros(28*28), {1, 28, 28});
auto target = make_input(zeros(10), {10});

// 3. Build the network
auto x = conv2d(input, conv1_kernel, 1, 8, 3, 3, 1, 0);
x = relu(x);
x = maxpool2d(x, 2, 2, 2);
// ... more layers
auto loss = softmax_cross_entropy_loss(logits, target);

// 4. Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (int i = 0; i < num_samples; i++) {
        // Set input data
        input->set_input(training_images[i]);
        target->set_input(training_labels[i]);
        
        // Forward pass
        loss->zero_grad_recursive();
        loss->calc();
        
        // Backward pass
        loss->backward();
        
        // Update parameters
        conv1_kernel->update(learning_rate);
        conv1_bias->update(learning_rate);
        fc_weight->update(learning_rate);
        // ... update all parameters
    }
}
```

## Available Operations

### CNN-Specific Operations

```cpp
// Convolution
auto output = conv2d(input, kernel, in_channels, out_channels, 
                     kernel_h, kernel_w, stride, padding);

// Max Pooling
auto pooled = maxpool2d(input, pool_h, pool_w, stride);

// Reshape
auto reshaped = reshape(input, {new_shape});

// Flatten
auto flat = flatten(input);

// Softmax + Cross Entropy Loss
auto loss = softmax_cross_entropy_loss(logits, targets);
```

### Activation Functions

```cpp
auto activated = relu(input);
auto activated = sigmoid(input);
auto activated = tanh_activation(input);
```

### Basic Operations

```cpp
auto sum = add(a, b);
auto diff = sub(a, b);
auto product = mul(a, b);  // Element-wise or matrix multiplication
auto powered = pow_elementwise(a, exponent);
```

### Loss Functions

```cpp
auto loss = mse_loss(predictions, targets);
auto loss = binary_cross_entropy_loss(predictions, targets);
auto loss = softmax_cross_entropy_loss(logits, targets);
```

## Training Tips

### Hyperparameters

```cpp
double learning_rate = 0.01;  // Start with 0.01
int num_epochs = 10;          // More for real datasets
int batch_size = 1;           // Framework limitation
```

### Weight Initialization

```cpp
// Use small random values
std::normal_distribution<double> dist(0.0, 0.1);
std::vector<double> weights;
for (int i = 0; i < size; i++) {
    weights.push_back(dist(gen));
}
```

### Monitoring Training

```cpp
if (epoch % 10 == 0) {
    std::cout << "Epoch " << epoch 
              << ", Loss: " << loss->item() 
              << std::endl;
}
```

## Documentation

- **Detailed CNN Documentation**: [docs/CNN_ARCHITECTURE.md](docs/CNN_ARCHITECTURE.md)
- **Implementation Details**: [docs/IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)
- **Main README**: [README.md](README.md)

## Common Issues

### Issue: Compilation errors

**Solution**: Ensure you're using C++23 or later:
```bash
g++ --version  # Should be 11 or higher
```

### Issue: Shape mismatch errors

**Solution**: Check that tensor shapes match at each layer:
```cpp
// Print shapes for debugging
std::cout << "Shape: ";
for (auto s : tensor->shape()) std::cout << s << " ";
std::cout << std::endl;
```

### Issue: Loss not decreasing

**Solution**: 
- Check learning rate (try 0.001 - 0.1)
- Verify data is normalized
- Ensure gradients are being computed

## Next Steps

1. **Load Real MNIST Data**: Replace synthetic data with actual MNIST images
2. **Train Longer**: Increase epochs to 20-50 for better accuracy
3. **Experiment**: Try different architectures (more layers, different sizes)
4. **Optimize**: Adjust hyperparameters for better performance

## Support

For issues or questions:
- Check the documentation in `docs/`
- Review example code in `test/`
- Open an issue on GitHub

## License

See [LICENSE](LICENSE) for details.
