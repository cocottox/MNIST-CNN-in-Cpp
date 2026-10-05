import gzip
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve


def test_reader():
    with open("MNIST_ORG/train-images.idx3-ubyte","rb") as f:
        magic = int.from_bytes(f.read(4),"big")
        num_images = int.from_bytes(f.read(4),"big")
        rows = int.from_bytes(f.read(4),"big")
        cols = int.from_bytes(f.read(4),"big")
        buf = f.read(rows*cols*num_images)
        data = np.frombuffer(buf,dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images,rows,cols,1)
        data = data / 255.0
        return data

def label_reader():
    with open("MNIST_ORG/train-labels.idx1-ubyte","rb") as f:
        magic = int.from_bytes(f.read(4),"big")
        num_labels = int.from_bytes(f.read(4),"big")
        buf = f.read(num_labels)
        labels = np.frombuffer(buf,dtype=np.uint8)
        return labels

def softmax(x):
    exp_x = np.exp(x-np.max(x))
    return exp_x / np.sum(exp_x)

def cross_entropy(probs,label):
    return -np.log(probs[label,0])

class ReLu:
    def __init__(self):
        self.X = None
    def forward(self,X):
        self.X = X
        return np.maximum(0,X)
    def backward(self,dY):
        dX = dY * (self.X > 0)
        return dX
    
class convolution:
    def __init__(self,n_filters,s_filters,input_c):
        self.f = s_filters
        self.n_filters = n_filters
        self.W = np.random.randn(n_filters,s_filters,s_filters,input_c)
        self.b = np.zeros(n_filters)
        self.X = None

    def forward(self,X):
        self.X = X
        H,W,C = X.shape
        h_out, w_out = H -self.f + 1, W - self.f + 1
        out = np.zeros((h_out,w_out,self.n_filters))
        for i in range(h_out):
            for j in range(w_out):
                X_slice = X[i:i+self.f,j:j+self.f,:]
                for k in range(self.n_filters):
                    out[i,j,k]=np.sum(X_slice*self.W[k])+self.b[k]
        return out

    def backward(self,dO,lr):
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(self.X)
        h_out, w_out, _ = dO.shape
        for i in range(h_out):
            for j in range(w_out):
                X_slice = self.X[i:i+self.f,j:j+self.f,:]
                for k in range(self.n_filters):
                    dW[k] += dO[i,j,k] * X_slice
                    db[k] += dO[i,j,k]
                    dX[i:i+self.f,j:j+self.f,:] += dO[i,j,k] * self.W[k]
        self.W -= lr*dW
        self.b -= lr*db
        return dX

class MaxPool:
    def forward(self,X):
        self.X = X
        H, W, C = X.shape
        out = np.zeros((H//2,W//2,C))
        for c in range(C):
            for i in range(H//2):
                for j in range(W//2):
                    out[i,j,c] = np.max(X[i*2:i*2+2,j*2:j*2+2,c])
        return out
    
    def backward(self,dO):
        dX = np.zeros_like(self.X)
        H, W, C = self.X.shape
        for c in range(C):
            for i in range(H//2):
                for j in range(W//2):
                    patch = self.X[i*2:i*2+2,j*2:j*2+2,c]
                    max_val = np.max(patch)
                    idx = np.argwhere(patch == max_val)[0]
                    dX[i*2 + idx[0],j*2+idx[1],c] = dO[i,j,c]
        return dX

class flatten:
    def forward(self,X):
        self.shape = X.shape
        return X.flatten().reshape(-1,1)
    def backward(self,dY):
        return dY.reshape(self.shape)
        
class connected_layer:
    def __init__(self,in_vector,out_vector):
        self.in_vector = in_vector
        self.out_vector = out_vector
        self.W = np.random.randn(out_vector,in_vector)*np.sqrt(2.0/in_vector)
        self.b = np.zeros((out_vector,1))
        self.X = None

    def forward(self,X):
        self.X = X.reshape(-1,1)
        Y = np.dot(self.W,self.X) + self.b
        return Y
    
    def backward(self,dY,learning_rate):
        dW = np.dot(dY,self.X.T)
        db = dY
        dX = np.dot(self.W.T,dY)
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return dX



if __name__ == "__main__":
    train = test_reader()
    labels = label_reader()
    conv = convolution(n_filters=4,s_filters=3,input_c=1)
    pool = MaxPool()
    flat = flatten()
    relu = ReLu()
    dence = connected_layer(in_vector=676,out_vector=10)
    learning_rate = 0.001
    epochs = 2
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = len(labels)
        for i in range(len(labels)):
            img = train[i]
            label = labels[i]

            out_conv = conv.forward(img)
            out_relu = relu.forward(out_conv)
            out_pool = pool.forward(out_relu)
            out_flat = flat.forward(out_pool)
            logits = dence.forward(out_flat)
            probs = softmax(logits)
            pred_one = np.zeros_like(probs)
            max_idx = np.argmax(probs)
            pred_one[max_idx,0] = 1.0


            loss = cross_entropy(probs,label)
            total_loss += loss

            if pred_one[label,0]== 1.0:
                correct += 1

            dY = probs.copy()
            dY[label,0] -= 1

            d_flat = dence.backward(dY,learning_rate)
            d_pool = flat.backward(d_flat)
            d_relu = pool.backward(d_pool)
            d_conv = relu.backward(d_relu)
            conv.backward(d_conv,learning_rate)
            if (i%100)==0 and i > 0:
                avg_correct = correct / i
                print(f"avg correct = ({avg_correct*100})% ---------- ({i}-{total})\n")
    np.savez(
            "cnn_MINST_model.npz",
            conv_W = conv.W,
            conv_b = conv.b,
            dense_W = dence.W,
            dense_b = dence.b
            )
    

        
