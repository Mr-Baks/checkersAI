import numpy as np
import json

class NN:
    def __init__(self, *layers):
        self.W = [np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2./layers[i]) for i in range(len(layers) - 1)]
        self.b = [np.zeros((1, layers[i + 1])) for i in range(len(layers) - 1)]
        self.a = []
        self.z = []
        self.learning_rate = 0.001

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X, training=False):
        self.a, self.z = [], []
        self.a.append(X) 
    
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z_current = np.dot(self.a[-1], W) + b
            
            if i == len(self.W) - 1:
                a_current = z_current
            else:
                a_current = self.relu(z_current)
            
            self.z.append(z_current)
            self.a.append(a_current)
    
        return self.a[-1]
    
    def backward(self, X, y):
        m = X.shape[0]  
        
        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        
        delta = (self.a[-1] - y) / m
        
        for i in range(len(self.W) - 1, -1, -1):
            dW[i] = np.dot(self.a[i].T, delta)
            db[i] = np.sum(delta, axis=0, keepdims=True)
            
            if i > 0:
                delta = np.dot(delta, self.W[i].T) * self.relu_derivative(self.z[i-1])
        
        for i in range(len(self.W)):
            self.W[i] -= self.learning_rate * dW[i]
            self.b[i] -= self.learning_rate * db[i]

    def train(self, X, y, epochs=1, learning_rate=0.001, batch_size=32, verbose=True):
        m = X.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_loss = 0
            num_batches = 0
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                output = self.forward(X_batch)
                
                loss = np.mean((output - y_batch) ** 2)
                total_loss += loss
                num_batches += 1
                
                self.backward(X_batch, y_batch, learning_rate)
            
            if verbose and epoch % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

    def predict(self, X):
        return self.forward(X)
    
    def save(self, filename='checkers_model.json'):
        params = {
            'W': [w.tolist() for w in self.W],
            'b': [b.tolist() for b in self.b],
            'architecture': [self.W[0].shape[0]] + [w.shape[1] for w in self.W]
        }
        with open(filename, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Model saved to {filename}")

    def load(self, filename='checkers_model.json'):
        try:
            with open(filename, 'r') as f:
                params = json.load(f)
                self.W = [np.array(w) for w in params['W']]
                self.b = [np.array(b) for b in params['b']]
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found. Using initialized weights.")
        except Exception as e:
            print(f"Error loading model: {e}")