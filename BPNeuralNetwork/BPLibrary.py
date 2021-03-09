# 导入相关库
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist

# 封装数据读取及预处理的代码
def load_data():
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    number=10000
    x_train=x_train[0:number]
    y_train=y_train[0:number]
    x_train=x_train.reshape(number,28*28)
    x_test=x_test.reshape(x_test.shape[0],28*28)
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    y_train=np_utils.to_categorical(y_train,10)
    y_test=np_utils.to_categorical(y_test,10)
    x_train=x_train
    x_test=x_test
    x_train=x_train/255
    x_test=x_test/255
    return (x_train,y_train),(x_test,y_test)


if __name__ == '__main__':

    # 调用方法
    (x_train,y_train),(x_test,y_test)=load_data()
    print(x_train.shape)
    print(x_test.shape)

    '''
    随便初始化一个NN模型
    '''
    model=Sequential()
    model.add(Dense(input_dim=28*28,units=633,activation='sigmoid'))
    model.add(Dense(units=633,activation='sigmoid'))
    model.add(Dense(units=633,activation='sigmoid'))
    model.add(Dense(units=10,activation='softmax'))

    model.compile(loss='mse',optimizer=SGD(lr=0.1),metrics=['accuracy'])

    model.fit(x_train,y_train,batch_size=100,epochs=20)

    train_result = model.evaluate(x_train,y_train,batch_size=10000)
    test_result = model.evaluate(x_test,y_test)

    print('TRAIN Accuracy:',train_result[1])
    print('TEST Accuracy:',test_result[1])