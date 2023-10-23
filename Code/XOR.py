import numpy as np

# I. Build Neural Network Architecture:
class Perceptron:
  # init NN:
  def __init__(self, n, alpha=0.1):
    self.W = np.random.randn(n+1, 1)  # Sửa lỗi ở đây, bỏ dấu ngoặc tròn
    self.alpha = alpha

  # Activation function:
  def step(self, x):
    return 1 if x > 0 else 0

  # training function:
  def fit(self, X, y, epochs=10):
    x = np.c_[X, np.ones((X.shape[0]))]
    print("shape của x:")
    print(x)
    for epoch in np.arange(0, epochs):
        for (x_i, target) in zip(x, y):
            p = self.step(np.dot(x_i, self.W))
            if p != target:
                error = p - target
                self.W += -self.alpha * error * x_i.reshape(3, 1)  # Sửa lỗi ở đây


  # prediction function:
  def predict(self, X, addBias=True):
    X = np.atleast_2d(X)
    if addBias:
      X = np.c_[X, np.ones((X.shape[0]))]
    return self.step(np.dot(X, self.W))


# Implement:

# 1. Initial dataset:
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y_or = np.array([[0],[1],[1],[1]])
y_and = np.array([[0],[0],[0],[1]])
y_xor = np.array([[0],[1],[1],[0]])
y_nor = np.array([[1],[0],[0],[0]])

# 2. Training:
print("train: ")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y_xor, epochs=20)

# 3. Evaluation
print("test: ")
for (x, target) in zip(X, y_xor):
  pred = p.predict(x)
  print("data={}, GT={}, pred={}".format(x, target[0], pred))
