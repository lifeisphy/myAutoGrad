import pandas as pd

import matplotlib.pyplot as plt

import cppyy
cppyy.include("autograd.hpp")
import cppyy.gbl as t
# Read the CSV file
df = pd.read_csv('testcases/digit-recognizer/train.csv')
fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
# Get the first row
n = 28
n_input = n * n
n_output = 10
n_kernel = 32
n_kernel_2 = 64
x = t.make_input([0 for _ in range(n_input)], [n,n])
label = t.make_input([0 for _ in range(n_output)], [n_output])
kernel_1 = t.make_param([0 for _ in range(3 * 3 * n_kernel)], [3, 3, n_kernel])
kernel_2 = t.make_param([0 for _ in range(3 * 3 * n_kernel_2 * n_kernel)], [3, 3,n_kernel, n_kernel_2])
output=[]
for i in range(n_kernel):
    a = t.slice(kernel_1, [-1,-1,i])
    b = t.conv2d(x,a)
    c= t.relu(b)
    res = t.MaxPooling(c)
    output.append(res)
res = t.stack(output) # 32, 13, 13

slices = [t.slice(res,[i,-1,-1]) for i in range(n_kernel)]
output = []
for i in range(n_kernel_2):
    lst=[]
    for j in range(n_kernel):
        a = t.slice(kernel_2, [-1,-1,j,i])
        b = t.conv2d(slices[j],a)
        lst.append(b)
    c = t.sum(lst)
    d = t.relu(c)
    output.append(d)
output = t.stack(output) #64, 11, 11

input_size = output.size()
mid_size = 128
W1 = t.make_param([0 for _ in range(input_size * mid_size)], [input_size, mid_size])
b1 = t.make_param([0 for _ in range(mid_size)], [mid_size])
W2 = t.make_param([0 for _ in range(mid_size * n_output)], [mid_size, n_output])
b2 = t.make_param([0 for _ in range(n_output)], [n_output])
layer1 = t.relu(t.mul(W1, output.flatten(),0,0) + b1 )
layer2 = t.relu(t.mul(W2, layer1,0,0) + b2 )
loss = t.cross_entropy(layer2, label)
print(output.shape())

inputs = [x, label]
params = [kernel_1, kernel_2, W1, b1, W2, b2]

# #todo: edit the training loop
# for epoch in range(100):
#     # for index, row in df.iterrows():
#     for i in range(60000):
#         row = df.iloc[i]
#         label_data = row['label']
#         pixels = row.drop('label').values.reshape(28, 28)
#         x.set_data(pixels.flatten().tolist())
#         label.set_data([1 if j == label_data else 0 for j in range(n_output)])
#         # 前向计算
#         loss.zero_grad_recursive()
#         loss.calc()
#         # if i % 1000 == 0:
#         print(f"Epoch {epoch}, Step {i}, Loss: {loss.data()}")
#         # 反向传播
#         loss.grad().set_data([1.0])  # 初始化输出节点的梯度为1
#         loss.backward()
#         # 更新参数
#         learning_rate = 0.001
#         for param in params:
#             if param.has_grad():
#                 grad = param.grad().data()
#                 data = param.data()
#                 updated_data = [d - learning_rate * g for d, g in zip(data, grad)]
#                 param.set_data(updated_data)
# x1 = t.conv2d(x, t.make_param([0 for _ in range(3*3)], [3,3]))
# x1p = t.relu(x1)
# for i in range(100):
#     row = df.iloc[i]
#     label = row['label']
#     pixels = row.drop('label').values.reshape(28, 28)
    
#     # Plot the image
#     ax[i // 10, i % 10].imshow(pixels, cmap='gray')
#     ax[i // 10, i % 10].set_title(f'Label: {label}',size=8)
#     ax[i // 10, i % 10].axis('off')
# plt.savefig(f'mnist_digits.png')