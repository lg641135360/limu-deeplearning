```python
class RNN:
    # ...
    def step(self,x):
        # update the hidden state  self.h存储上一时刻的隐层输出（内存存储）
        self.h = np.tanh(np.dot(self.W_hh,self.h) + np.dot(self.W_xh,x))
        # compare the output vector
        y = np.dot(selft.W_hy,self.h)
        return y
```

* 外面需要进行一个不定长的循环

* 训练起来极为复杂