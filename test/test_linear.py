import cppyy
cppyy.include("autograd.hpp")
import cppyy.gbl as t
import numpy as np
import random
n_samples = 100
n_features = 3
n_outputs = 2
w0 = np.random.randn(n_features,n_outputs)
b0 = np.random.randn(n_outputs)
# f(x) = w0 x + b0 + noise
X = np.random.randn(n_samples,n_features)
Y = np.dot(X,w0) + b0 # 0.1 * np.random.randn(n_samples,n_outputs)
X = X.tolist()
Y = Y.tolist()

x = t.make_input(list(range(n_features)))
y = t.make_input(list(range(n_outputs)))
w = t.make_param(np.random.randn(n_features,n_outputs).flatten().tolist(),[n_features, n_outputs])
b = t.make_param(np.random.randn(n_outputs).flatten().tolist(),[n_outputs])
wx = t.mul(w,x,0,0)
yhat = t.add(wx,b)
loss = t.mse_loss(yhat,y)
for epoch in range(100):
    total_loss = 0
    print(f"epoch {epoch}:")
    for xi,yi in zip(X,Y):
        loss.zero_grad_recursive()
        x.set_input(xi)
        y.set_input(yi)
        loss.calc()
        loss.backward()
        total_loss += loss.item()
        for p in [w,b]:
            p.update(0.01)
    print(f"loss {total_loss:.4f}")
print("end:")
print("w:", np.array(w.data()).reshape(n_features,n_outputs))
print("b:", np.array(b.data()).reshape(n_outputs))
print("w0:", w0)
print("b0:", b0)