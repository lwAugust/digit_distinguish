# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from chapter4.allfun import TwoLayerNet
import random
from pylab import mpl
import glob
import imageio

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 读入数据，分别表示 用于训练的数据  用于训练的数据对应的标签 用于测试的数据 用于测试的数据对应的标签
# 暂时没必要知道它是怎么加载的，你知道知道他们表示什么就可以了
# #另外，图片数据是（28*28）像素的矩阵，修改load_mnist的参数也可以把（28*28）转换成（784,）这样的形状
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 这个network 就是我们的神经网络
network = TwoLayerNet(input_size=784, hidden_size=20, output_size=10)


iters_num = 5000  # 适当设定循环的次数

# 每次从所有训练数据中抽出100个来学习  这个叫做mini-batch学习
batch_size = 100

# 一共多少个训练数据 60000个
train_size = x_train.shape[0]


# 学习率，这个影响每次修正神经网络中参数时的力度
learning_rate = 0.01

# 训练开始
for i in range(iters_num):
    # mini - batch学习，随机抽取 batch_size 个数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度 下面是两种方法，还记得梯度是干嘛的吗，它告诉我如何能让我的loss减小的最快
    # 第一种方法较慢，叫做数值微分，就是用python计算微分结果
    # 第二种方法快 叫做误差反向传播，我觉得自己无法解释清楚，它是一种利用"计算图"来快速求微分的方法
    # 返回的这个值，用于更新神经网络的参数

    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 更新参数，一共4个参数，每层神经网络2个，一个2个神经网络层
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]


print('模型训练完成')


# n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

# # 修改这个值，测试不同图片
# whtch = 123
# y = network.predict(x_test[whtch])
# print('y ', y)
# y = np.argmax(y)
#
# mytestone = np.array(x_test[whtch])
# mytestone = mytestone.reshape(28, 28)
#
# plt.imshow(mytestone, cmap='gray')
# plt.title('识别结果：'+str(y))
# plt.show()

# our own image test data set
our_own_dataset = []

# load the png image data as test data set
for image_file_name in glob.glob('E:\\Work\\Python\\python_prj\\pure_projs\\deep-learning\\self_5.png'):
    # use the filename to set the correct label
    label = int(image_file_name[-5:-4])
    print("label ... ", label)
    # load image data from png files into an array
    print("loading ... ", image_file_name)
    img_array = imageio.imread(image_file_name, as_gray=True)

    # reshape from 28x28 to list of 784 values, invert values
    img_data = 255.0 - img_array.reshape(784)

    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(np.min(img_data))
    print(np.max(img_data))

    # append label and image data  to test data set
    record = np.append(label, img_data)
    our_own_dataset.append(record)

    pass

print(our_own_dataset)
# test the neural network with our own images

# record to test
item = 0

# plot image
plt.imshow(our_own_dataset, cmap='Greys', interpolation='None')

# correct answer is first value
correct_label = our_own_dataset[item][0]
# data is remaining values
inputs = our_own_dataset[item][1:]

# query the network
outputs = network.query(inputs)
# print(outputs)

# the index of the highest value corresponds to the label
label = np.argmax(outputs)
print("network says ", label)
# append correct or incorrect to list
if label == correct_label:
    print("Good,match!")
else:
    print("no match!")
    pass
