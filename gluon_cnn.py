# -*- coding: utf-8 -*-

from mxnet import nd
import mxnet as mx
import mxnet.gluon as gluon
from mxnet.gluon import nn
import mxnet.autograd as autograd
import sys
sys.path.append('..')
import utils
from utils import load_data_fashion_mnist


# 创建网络
class CNN(nn.Block):
    def __init__(self, **kwargs):
        super(CNN, self).__init__(**kwargs)
        with self.name_scope():
            self.conv0 = nn.Conv2D(channels=64, kernel_size=5, activation='relu')
            self.conv1 = nn.Conv2D(channels=128, kernel_size=3, activation='relu')
            self.maxpool = nn.MaxPool2D(pool_size=2, strides=2)
            self.flatten = nn.Flatten()
            self.dense0 = nn.Dense(256, activation='relu')
            self.dense1 = nn.Dense(10)

    def forward(self, x):
        x = self.conv0(x)
        # x = self.maxpool(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense0(x)
        out = self.dense1(x)
        return out


# 实例化网络、gpu、损失函数、优化器
net = CNN()
ctx = utils.try_gpu()
net.initialize(ctx=ctx)
cross_loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), "adam", {'learning_rate': 0.1})

# 准备数据
batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size, root='./fashion-mnist')

# 训练
epoches = 10
for e in range(epoches):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = cross_loss(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net, ctx=ctx)

    print("%d epoach: the train loss is %f, the train accracy is %f, the test accuracy is %f" % (
    e, train_loss / len(train_data), train_acc / len(train_data), test_acc))
