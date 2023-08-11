#単一ニューロンの実装
import numpy as np
import matplotlib.pyplot as plt
#X = np.arange(-1.0, 1.0, 0.2)
#Y = np.arange(-1.0, 1.0, 0.2)
#Z = np.zeros((10,10))
#w_x = 2.5
#w_y = 3.0
#bias = 0.1
#for i in range(10):
#    for j in range(10):
#        u = X[i]*w_x + Y[j]*w_y + bias
#        y = 1/(1+np.exp(-u))
#        Z[j][i] = y
#plt.imshow(Z, "gray", vmin = 0.0, vmax = 1.0)
#plt.colorbar()
#plt.show()

def one_Newron(w_x, w_y, bias):
    X = np.arange(-1.0, 1.0, 0.2)
    Y = np.arange(-1.0, 1.0, 0.2)
    Z = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            u = X[i]*w_x + Y[j]*w_y + bias
            y = 1/(1+np.exp(-u))
            Z[j][i] = y
    plt.imshow(Z, "gray", vmin = 0.0, vmax = 1.0)
    plt.colorbar()
    plt.show()

#one_Neoron(2.5, 3.0, 0.1)
#one_Newron(-2.5, -3.0, 0.1)
#one_Newron(0, 3.0, 0.1)
#one_Newron(2.5, 0, 0.1)
#one_Newron(2.5, 3.0, -2.0)
#one_Newron(2.5, 3.0, 0)
#one_Newron(2.5, 3.0, 2.0)

#ニューラルネットワーク（回帰）
import numpy as np
import matplotlib.pyplot as plt
def newralnetwork3(w_im, w_mo, b_im, b_mo):
    X = np.arange(-1.0, 1.0, 0.2)
    Y = np.arange(-1.0, 1.0, 0.2)
    Z = np.zeros((10, 10))
#    w_im = np.array([[4.0,4.0],
#                     [4.0,4.0]])
#    w_mo = np.array([[1.0],
#                     [-1.0]])
#    b_im = np.array([3.0, -3.0])
#    b_mo = np.array([0.1])
    def middle_layer(x, w, b):
        u = np.dot(x, w) + b
        return 1/(1+np.exp(-u))
    def output_layer(x,w,b):
        u = np.dot(x, w) + b
        return u
    for i in range(10):
        for j in range(10):
            inp = np.array([X[i], Y[j]])
            mid = middle_layer(inp, w_im, b_im)
            out = output_layer(mid, w_mo, b_mo)
            Z[j][i] = out[0]
    plt.imshow(Z, "gray", vmin = 0.0, vmax = 1.0)
    plt.colorbar()
    plt.show()

#newralnetwork3(np.array([[4.0,4.0],[4.0,4.0]]), np.array([[1.0],[-1.0]]), b_im = np.array([3.0, -3.0]), b_mo = np.array([0.1]))
#newralnetwork3(np.array([[-5.0,-5.0],[5.0,-5.0]]), np.array([[1.0],[-1.0]]), np.array([0.0,0.0]), np.array([0.0]))
#newralnetwork3(np.array([[4.0,4.0],[4.0,4.0]]), np.array([[-1.0],[1.0]]), np.array([3.0,-3.0]), np.array([1.0]))
#newralnetwork3(np.array([[-4.0,4.0],[-4.0,-4.0]]), np.array([[1.0],[-1.0]]), np.array([3.0,-3.0]), np.array([0.0]))

#分類問題
def newralnetwork_classify(w_im, w_mo, b_im, b_mo):
    X = np.arange(-1.0, 1.0, 0.1)
    Y = np.arange(-1.0, 1.0, 0.1)
#    w_im = np.array([[1.0,2.0],
#                     [2.0,3.0]])
#    w_mo = np.array([[-1.0,1.0],
#                     [1.0,-1.0]])
#    b_im = no.array([0.3,-0.3])
#    b_mo = np.array([0.4,0.1])
    def middle_layer(x, w, b):
        u = np.dot(x, w) + b
        return 1/(1+np.exp(-u))
    def output_layer(x, w, b):
        u = np.dot(x, w) + b
        return np.exp(u)/np.sum(np.exp(u))
    x_1 = []
    y_1 = []
    x_2 = []
    y_2 = []
    for i in range(20):
        for j in range(20):
            inp = np.array([X[i], Y[j]])
            mid = middle_layer(inp, w_im, b_im)
            out = output_layer(mid, w_mo, b_mo)
            if out[0] > out[1]:
                x_1.append(X[i])
                y_1.append(Y[j])
            else:
                x_2.append(X[i])
                y_2.append(Y[j])
    plt.scatter(x_1, y_1, marker="+")
    plt.scatter(x_2, y_2, marker="o")
    plt.show()


newralnetwork_classify(np.array([[1.0,2.0],[2.0,3.0]]), np.array([[-1.0,1.0],[1.0,-1.0]]), np.array([0.3,-0.3]), np.array([0.4,0.1]))
newralnetwork_classify(np.array([[2.0,1.0],[0.0,3.0]]), np.array([[-2.0,1.0],[-1.0,1.0]]), np.array([-0.3,-0.3]), np.array([0.4,-1.2]))
newralnetwork_classify(np.array([[2.0,2.0],[2.0,3.0]]), np.array([[-1.0,1.0],[1.0,-1.0]]), np.array([0.3,-0.3]), np.array([0.4,0.1]))
