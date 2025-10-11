# CNN Architecture for MNIST Dataset

This document describes the CNN architecture implementation for the MNIST digit classification task using the myAutoGrad framework.

## Architecture Overview

The CNN architecture consists of the following layers:

```
Input (28x28x1)
    ↓
Conv2D (1→8 channels, 3x3 kernel, stride=1) → 26x26x8
    ↓
ReLU Activation
    ↓
MaxPool2D (2x2, stride=2) → 13x13x8
    ↓
Conv2D (8→16 channels, 3x3 kernel, stride=1) → 11x11x16
    ↓
ReLU Activation
    ↓
MaxPool2D (2x2, stride=2) → 5x5x16
    ↓
Flatten → 400
    ↓
Dense (400→128)
    ↓
ReLU Activation
    ↓
Dense (128→10)
    ↓
Softmax + Cross Entropy Loss
```

## Parameter Count

- **Conv1 Layer**: 8 × 1 × 3 × 3 + 8 = 80 parameters
- **Conv2 Layer**: 16 × 8 × 3 × 3 + 16 = 1,168 parameters
- **FC1 Layer**: 128 × 400 + 128 = 51,328 parameters
- **FC2 Layer**: 10 × 128 + 10 = 1,290 parameters
- **Total**: 53,866 parameters

## New Operations Added

### 1. **reshape()**
Changes the shape of a tensor without modifying the data.

```cpp
VarPtr reshape(VarPtr a, const std::vector<size_t>& new_shape);
```

**Example:**
```cpp
auto x = make_var({1, 2, 3, 4, 5, 6}, {2, 3});
auto y = reshape(x, {3, 2});  // Reshape from [2, 3] to [3, 2]
```

### 2. **flatten()**
Converts a multi-dimensional tensor to a 1D vector.

```cpp
VarPtr flatten(VarPtr a);
```

**Example:**
```cpp
auto x = make_var({1, 2, 3, 4}, {2, 2});
auto y = flatten(x);  // Output shape: [4]
```

### 3. **conv2d()**
Performs 2D convolution operation.

```cpp
VarPtr conv2d(VarPtr input, VarPtr kernel, 
              size_t in_channels, size_t out_channels,
              size_t kernel_h, size_t kernel_w, 
              size_t stride = 1, size_t padding = 0);
```

**Parameters:**
- `input`: Input tensor with shape [in_channels, height, width]
- `kernel`: Convolution kernel with shape [out_channels, in_channels, kernel_h, kernel_w]
- `stride`: Stride for convolution (default: 1)
- `padding`: Zero padding (default: 0)

**Example:**
```cpp
auto input = make_input(std::vector<double>(1*28*28, 0.0), {1, 28, 28});
auto kernel = make_param(random_normal(8*1*3*3), {8, 1, 3, 3});
auto output = conv2d(input, kernel, 1, 8, 3, 3, 1, 0);  // Output: [8, 26, 26]
```

### 4. **maxpool2d()**
Performs 2D max pooling operation.

```cpp
VarPtr maxpool2d(VarPtr input, size_t pool_h, size_t pool_w, size_t stride = 2);
```

**Parameters:**
- `input`: Input tensor with shape [channels, height, width]
- `pool_h`, `pool_w`: Pooling window size
- `stride`: Stride for pooling (default: 2)

**Example:**
```cpp
auto x = make_var(data, {8, 26, 26});
auto pooled = maxpool2d(x, 2, 2, 2);  // Output: [8, 13, 13]
```

### 5. **softmax_cross_entropy_loss()**
Computes softmax activation followed by cross-entropy loss.

```cpp
VarPtr softmax_cross_entropy_loss(VarPtr logits, VarPtr targets);
```

**Parameters:**
- `logits`: Raw output scores (before softmax)
- `targets`: One-hot encoded target labels

**Example:**
```cpp
auto logits = make_var({0.1, 0.5, 0.2, 0.8, 0.3, 0.6, 0.4, 0.7, 0.9, 0.0}, {10});
auto targets = make_var({0, 0, 0, 1, 0, 0, 0, 0, 0, 0}, {10});  // One-hot for class 3
auto loss = softmax_cross_entropy_loss(logits, targets);
```

## Usage Example

### C++ Implementation

See the complete working example in `test/mnist_cnn_demo.cpp`:

```bash
# Compile
g++ -std=c++23 -Wall -Wextra -O2 -o test/mnist_cnn_demo test/mnist_cnn_demo.cpp

# Run
./test/mnist_cnn_demo
```

### Python Implementation (with cppyy)

See `test/mnist_cnn.py` for a Python wrapper example using cppyy:

```bash
# Run (requires cppyy)
python3 test/mnist_cnn.py
```

## Training Tips

### Hyperparameters

- **Learning Rate**: 0.01 is a good starting point
- **Batch Size**: Due to framework limitations, process samples individually
- **Epochs**: 10-20 epochs for small datasets, more for full MNIST
- **Initialization**: Use small random values (mean=0, std=0.05-0.1)

### Expected Performance

With proper training on the full MNIST dataset:
- Training time: ~5-10 minutes per epoch (depends on hardware)
- Expected accuracy: 95-98%
- Loss: Should decrease from ~2.3 to < 0.1

## Architecture Variations

### Lighter Model (Fewer Parameters)

```cpp
Conv2D (1→4 channels, 3x3) → MaxPool2D → 
Conv2D (4→8 channels, 3x3) → MaxPool2D → 
Flatten → Dense(400→64) → Dense(64→10)
```

Total parameters: ~13,500

### Deeper Model (Better Accuracy)

```cpp
Conv2D (1→16 channels, 3x3) → ReLU → MaxPool2D →
Conv2D (16→32 channels, 3x3) → ReLU → MaxPool2D →
Conv2D (32→64 channels, 3x3) → ReLU → MaxPool2D →
Flatten → Dense(256→128) → ReLU → Dense(128→10)
```

Total parameters: ~150,000

## Implementation Notes

1. **Shape Preservation**: All activation functions (ReLU, Sigmoid, Tanh) now preserve the input tensor shape.

2. **Broadcasting Limitation**: The current framework has limited broadcasting support. Biases need to be manually expanded to match the feature map dimensions.

3. **Memory Efficiency**: The framework uses forward and backward functions that are called during the forward and backward passes, allowing for efficient computation graph execution.

4. **Gradient Computation**: All CNN operations fully support automatic differentiation with correct gradient computation.

## Testing

The implementation includes comprehensive tests:

```bash
# Run the CNN demo
make mnist_cnn_demo
./test/mnist_cnn_demo

# Expected output:
# - Architecture visualization
# - Parameter count
# - Training progress over 3 epochs
# - Loss decreasing over time
```

## Future Improvements

Potential enhancements for the CNN implementation:

1. **Batch Processing**: Support for batch dimensions
2. **Batch Normalization**: Add batch norm layers
3. **Dropout**: Implement dropout for regularization
4. **Different Pooling**: Add average pooling, adaptive pooling
5. **Padding Modes**: Support different padding strategies
6. **Dilated Convolutions**: Add support for dilation
7. **Data Augmentation**: Image transformations for training
8. **Advanced Optimizers**: Adam, RMSprop, etc.

## References

- LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998)
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- CS231n: Convolutional Neural Networks for Visual Recognition

## License

This implementation is part of the myAutoGrad framework. See the main repository LICENSE for details.
