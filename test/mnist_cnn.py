"""
CNN Architecture for MNIST Dataset using myAutoGrad framework
This script demonstrates a simple CNN for digit classification
"""

import cppyy
cppyy.include("autograd.hpp")
import cppyy.gbl as ag
import struct
import gzip
import os
import urllib.request

def download_mnist():
    """Download MNIST dataset if not already present"""
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    data_dir = 'mnist_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
            print(f"Downloaded {filename}")
    
    return data_dir

def load_mnist_images(filename):
    """Load MNIST images from file"""
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = []
        for _ in range(num):
            image = struct.unpack(f">{rows*cols}B", f.read(rows * cols))
            # Normalize to [0, 1]
            image = [float(pixel) / 255.0 for pixel in image]
            images.append(image)
        return images, rows, cols

def load_mnist_labels(filename):
    """Load MNIST labels from file"""
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = struct.unpack(f">{num}B", f.read(num))
        return list(labels)

def label_to_onehot(label, num_classes=10):
    """Convert label to one-hot encoding"""
    onehot = [0.0] * num_classes
    onehot[label] = 1.0
    return onehot

class SimpleCNN:
    """
    Simple CNN Architecture for MNIST:
    - Input: 28x28 grayscale images
    - Conv1: 1 -> 8 channels, 3x3 kernel, stride=1, padding=0 -> 26x26
    - ReLU
    - MaxPool: 2x2, stride=2 -> 13x13
    - Conv2: 8 -> 16 channels, 3x3 kernel, stride=1, padding=0 -> 11x11
    - ReLU
    - MaxPool: 2x2, stride=2 -> 5x5
    - Flatten: 16*5*5 = 400
    - FC1: 400 -> 128
    - ReLU
    - FC2: 128 -> 10 (output classes)
    - Softmax + Cross Entropy Loss
    """
    
    def __init__(self):
        # Initialize parameters with small random values
        import random
        random.seed(42)
        
        # Conv1: 1 input channel, 8 output channels, 3x3 kernel
        conv1_size = 8 * 1 * 3 * 3  # out_ch * in_ch * kh * kw
        self.conv1_kernel = ag.make_param(
            [random.gauss(0, 0.1) for _ in range(conv1_size)],
            [8, 1, 3, 3]
        )
        self.conv1_bias = ag.make_param([0.0] * 8, [8])
        
        # Conv2: 8 input channels, 16 output channels, 3x3 kernel
        conv2_size = 16 * 8 * 3 * 3
        self.conv2_kernel = ag.make_param(
            [random.gauss(0, 0.1) for _ in range(conv2_size)],
            [16, 8, 3, 3]
        )
        self.conv2_bias = ag.make_param([0.0] * 16, [16])
        
        # FC1: 400 -> 128
        fc1_size = 128 * 400
        self.fc1_weight = ag.make_param(
            [random.gauss(0, 0.05) for _ in range(fc1_size)],
            [128, 400]
        )
        self.fc1_bias = ag.make_param([0.0] * 128, [128])
        
        # FC2: 128 -> 10
        fc2_size = 10 * 128
        self.fc2_weight = ag.make_param(
            [random.gauss(0, 0.05) for _ in range(fc2_size)],
            [10, 128]
        )
        self.fc2_bias = ag.make_param([0.0] * 10, [10])
        
        # Create input and target placeholders
        self.input = ag.make_input([0.0] * (28 * 28), [1, 28, 28])
        self.target = ag.make_input([0.0] * 10, [10])
        
        # Build the network
        self._build_network()
    
    def _build_network(self):
        """Build the CNN computation graph"""
        # Conv1 layer
        x = self.input
        x = ag.conv2d(x, self.conv1_kernel, 1, 8, 3, 3, 1, 0)  # 26x26x8
        # Add bias (broadcast across spatial dimensions)
        x = ag.reshape(x, [8 * 26 * 26])
        bias1_expanded = ag.make_input([0.0] * (8 * 26 * 26), [8 * 26 * 26])
        x = ag.add(x, bias1_expanded)
        x = ag.reshape(x, [8, 26, 26])
        x = ag.relu(x)
        x = ag.maxpool2d(x, 2, 2, 2)  # 13x13x8
        
        # Conv2 layer
        x = ag.conv2d(x, self.conv2_kernel, 8, 16, 3, 3, 1, 0)  # 11x11x16
        x = ag.reshape(x, [16 * 11 * 11])
        bias2_expanded = ag.make_input([0.0] * (16 * 11 * 11), [16 * 11 * 11])
        x = ag.add(x, bias2_expanded)
        x = ag.reshape(x, [16, 11, 11])
        x = ag.relu(x)
        x = ag.maxpool2d(x, 2, 2, 2)  # 5x5x16
        
        # Flatten
        x = ag.flatten(x)  # 400
        
        # FC1
        x = ag.mul(self.fc1_weight, x, 1, 0)  # Matrix multiplication
        x = ag.add(x, self.fc1_bias)
        x = ag.relu(x)
        
        # FC2
        x = ag.mul(self.fc2_weight, x, 1, 0)
        logits = ag.add(x, self.fc2_bias)
        
        # Loss
        self.logits = logits
        self.loss = ag.softmax_cross_entropy_loss(logits, self.target)
        
        # Store intermediate variables for bias updates
        self.bias1_expanded = bias1_expanded
        self.bias2_expanded = bias2_expanded
    
    def get_parameters(self):
        """Return all trainable parameters"""
        return [
            self.conv1_kernel, self.conv1_bias,
            self.conv2_kernel, self.conv2_bias,
            self.fc1_weight, self.fc1_bias,
            self.fc2_weight, self.fc2_bias
        ]
    
    def forward(self, image, label):
        """Forward pass with given image and label"""
        # Set input image
        self.input.set_input(image)
        
        # Set target
        target_onehot = label_to_onehot(label)
        self.target.set_input(target_onehot)
        
        # Set bias values (need to expand for broadcasting)
        # This is a workaround since we don't have native broadcasting in add
        conv1_bias_data = list(self.conv1_bias.data())
        bias1_full = []
        for c in range(8):
            bias1_full.extend([conv1_bias_data[c]] * (26 * 26))
        self.bias1_expanded.set_input(bias1_full)
        
        conv2_bias_data = list(self.conv2_bias.data())
        bias2_full = []
        for c in range(16):
            bias2_full.extend([conv2_bias_data[c]] * (11 * 11))
        self.bias2_expanded.set_input(bias2_full)
        
        # Compute forward pass
        self.loss.calc()
        
        return self.loss.item()
    
    def backward(self):
        """Backward pass to compute gradients"""
        self.loss.backward()
    
    def update(self, learning_rate):
        """Update parameters using gradient descent"""
        for param in self.get_parameters():
            param.update(learning_rate)
    
    def zero_grad(self):
        """Zero out all gradients"""
        self.loss.zero_grad_recursive()
    
    def predict(self, image):
        """Predict class for a given image"""
        # Set input
        self.input.set_input(image)
        
        # Set bias values
        conv1_bias_data = list(self.conv1_bias.data())
        bias1_full = []
        for c in range(8):
            bias1_full.extend([conv1_bias_data[c]] * (26 * 26))
        self.bias1_expanded.set_input(bias1_full)
        
        conv2_bias_data = list(self.conv2_bias.data())
        bias2_full = []
        for c in range(16):
            bias2_full.extend([conv2_bias_data[c]] * (11 * 11))
        self.bias2_expanded.set_input(bias2_full)
        
        # Forward pass
        self.logits.calc()
        
        # Get predictions
        logits_data = list(self.logits.data())
        return logits_data.index(max(logits_data))


def train_cnn():
    """Train the CNN on MNIST dataset"""
    print("="*60)
    print("CNN Architecture for MNIST Dataset")
    print("="*60)
    
    # Download and load MNIST data
    print("\n1. Loading MNIST dataset...")
    try:
        data_dir = download_mnist()
        
        train_images, h, w = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
        train_labels = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
        test_images, _, _ = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
        test_labels = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
        
        print(f"   Training samples: {len(train_images)}")
        print(f"   Test samples: {len(test_images)}")
        print(f"   Image size: {h}x{w}")
    except Exception as e:
        print(f"   Error loading MNIST: {e}")
        print("   Using dummy data for demonstration...")
        # Create dummy data for demonstration
        train_images = [[0.0] * 784 for _ in range(100)]
        train_labels = [i % 10 for i in range(100)]
        test_images = [[0.0] * 784 for _ in range(20)]
        test_labels = [i % 10 for i in range(20)]
    
    # Initialize CNN
    print("\n2. Initializing CNN model...")
    print("   Architecture:")
    print("   - Input: 28x28 grayscale image")
    print("   - Conv1: 1->8 channels, 3x3 kernel -> 26x26")
    print("   - ReLU + MaxPool 2x2 -> 13x13")
    print("   - Conv2: 8->16 channels, 3x3 kernel -> 11x11")
    print("   - ReLU + MaxPool 2x2 -> 5x5")
    print("   - Flatten -> 400")
    print("   - FC1: 400->128")
    print("   - ReLU")
    print("   - FC2: 128->10")
    print("   - Softmax + Cross Entropy Loss")
    
    model = SimpleCNN()
    
    # Training parameters
    learning_rate = 0.01
    num_epochs = 5
    batch_size = 10  # Process samples one at a time due to framework limitations
    
    print(f"\n3. Training Configuration:")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Training samples per epoch: {min(batch_size, len(train_images))}")
    
    # Training loop
    print("\n4. Training Progress:")
    print("-"*60)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_samples = min(batch_size, len(train_images))
        
        for i in range(num_samples):
            # Zero gradients
            model.zero_grad()
            
            # Forward pass
            loss = model.forward(train_images[i], train_labels[i])
            total_loss += loss
            
            # Backward pass
            model.backward()
            
            # Update parameters
            model.update(learning_rate)
            
            if (i + 1) % 5 == 0:
                print(f"   Epoch {epoch+1}/{num_epochs}, Sample {i+1}/{num_samples}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_samples
        print(f"   Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        print("-"*60)
    
    # Test accuracy
    print("\n5. Evaluating on test set...")
    correct = 0
    num_test_samples = min(10, len(test_images))
    
    for i in range(num_test_samples):
        pred = model.predict(test_images[i])
        if pred == test_labels[i]:
            correct += 1
        print(f"   Sample {i+1}: Predicted={pred}, Actual={test_labels[i]}, {'✓' if pred == test_labels[i] else '✗'}")
    
    accuracy = 100.0 * correct / num_test_samples
    print(f"\n   Test Accuracy: {accuracy:.2f}% ({correct}/{num_test_samples})")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    try:
        train_cnn()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: This is a demonstration of the CNN architecture.")
        print("Full training may require adjustments for the custom autograd framework.")
