#ステップ関数の定義と実行
import numpy as np
import matplotlib.pyplot as plt
def step_function(x):
    return np.where(x<=0,0,1)
x = np.linspace(-5, 5)
y = step_function(x)
plt.plot(x,y)
plt.show()

#シグモイド関数の定義と実行
import numpy as np
import matplotlib.pylab as plt
def sigmoid_function(x):
    return 1/(1+np.exp(-x))
x = np.linspace(-5, 5)
y = sigmoid_function(x)
plt.plot(x, y)
plt.show()

#tanhの実装と実行
import numpy as np
import matplotlib.pylab as plt
def tanh_function(x):
    return np.tanh(x)
x = np.linspace(-5, 5)
y = tanh_function(x)
plt.plot(x, y)
plt.show()

#ReLUの定義と実行
import numpy as np
import matplotlib.pylab as plt
def relu_function(x):
    return np.where(x <= 0, 0, x)
x = np.linspace(-5, 5)
y = relu_function(x)
plt.plot(x, y)
plt.show()

#Leaky ReLUの定義と実行
import numpy as np
import matplotlib.pylab as plt
def leaky_relu_function(x):
    return np.where(x <= 0, 0.01*x, x)
x = np.linspace(-5, 5)
y = leaky_relu_function(x)
plt.plot(x, y)
plt.show()

#恒等関数の定義と実行
import numpy as np
import matplotlib.pylab as plt
x = np.linspace(-5, 5)
y = x
plt.plot(x,y)
plt.show()

#ソフトマックス関数の定義と実行
import numpy as np
import matplotlib.pylab as plt
def softmax_function(x):
    return np.exp(x)/np.sum(np.exp(x))
y = softmax_function(np.array([1, 2, 3]))
print(y)
