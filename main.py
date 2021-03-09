import tensorflow.keras as keras
import numpy as np
import os

class TrainingSets():
    """
    Methods that return [x], [y] arrays to train model on, and x/y domain for processing if needed
    """

    @staticmethod
    def MnistImageClassification():
        """
        Mnist number classification
        Returns image data (28,28,1), [0, 255] and
        corresponding (10x1), [0,1] labels.
        """
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        x = x_train
        y = y_train

        return np.array(x), np.array(y), [0,255], [0,1]

    @staticmethod
    def TextClassification():
        """
        Text sentiment classification
        Returns (512,1), [0,128], ascci representation of 
        text (0 being no text) and (1,1), [0,1] classification label.
        """
        def tokenise(string, max_length=512):
            v = []
            for char_ in string[:max_length]:
                #print(ord(char_))
                v.append(ord(char_))
            
            if len(string)<512: 
                for i in range(512-len(string)):
                    v.append(0)
            return v

        good = open('datasets/TextClassification/0.csv', 'r').readlines()
        bad  = open('datasets/TextClassification/1.csv', 'r').readlines()
        data = []
        for line in good:
            url = line.replace('\n', '')
            data.append([tokenise(url), 0])
        for line in bad:
            url = line.replace('\n', '')
            data.append([tokenise(url), 1])
        
        import random
        random.shuffle(data)

        x, y = [], []
        for i in data:
            x.append(i[0])
            y.append(i[1])
        
        x, y = np.array(x), np.array(y)
        x, y = np.reshape(x, (x.shape[0], 512,1)), np.reshape(y, (y.shape[0],1)) 

        return x,y, [0,128], [0,1]

    @staticmethod
    def UpScaler():
        """
        Mnist number upscaling
        Returns scaled image (8,8,1), [0, 255] and
        corresponding (28,28,1), [0,255] full-sized images.
        """
        import cv2
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        y = x_train
        x= []
        for image in y:
            image = np.array(image)
            scaled_image = cv2.resize(image, dsize=(8, 8), interpolation=cv2.INTER_CUBIC)
            scaled_image = np.reshape(scaled_image, (8,8,1))
            x.append(scaled_image)

        return np.array(x), np.array(y), [0,255], [0,255]
    
    @staticmethod
    def Regression():
        """
        Stock price prediction
        Returns data (32,1), [0, -1] and
        (1), [0,-1] price after. 
        """
        files = [f for f in os.listdir('datasets/Regression/') if os.path.isfile(os.path.join('datasets/Regression/', f))]
        data = []
        for filename in files:
            single_stock_data = []
            f = open('datasets/Regression/'+filename, 'r')
            lines = f.readlines()[1:]
            for line in lines:
                line = line.replace('\n', '').split(',')
                single_stock_data.append([ float(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8]), float(line[9]) ])
            data.append(single_stock_data)

        x, y = [], []
        for stock in data:
            for i in range(len(stock)-33):
                x.append(stock[i:i+33])
                y.append(stock[33])
        
        return np.array(x), np.array(y), [0,-1], [0,-1]

class Trainer():
    def __init__(self, model, processing):
        
        method_list = [func for func in dir(TrainingSets) if callable(getattr(TrainingSets, func)) and not func.startswith("__")]
        results = {}
        
        for set_ in method_list: 
            print(set_)
            x, y, x_domain, y_domain = getattr(TrainingSets, set_)()
            x, y = processing(x, y, x_domain, y_domain)
            
            current_model = self.AddInOutModel(model, x.shape[1:], y.shape[1:])
            current_model.fit(x, y, epochs=10, batch_size=512, validation_split=0.3, shuffle=True)

    def AddInOutModel(self, model, in_size, out_size):
        print(in_size)
        in_size = list(in_size)
        in_size.append(1)
        in_size = tuple(in_size)

        input_ = keras.layers.Input(shape=in_size)
        inter_ = model(input_)
        outer_ = keras.layers.Dense(np.prod(list(out_size)))(inter_)
        outer_ = keras.layers.Reshape((out_size))(outer_)

        full_model = keras.models.Model(inputs=input_, outputs=outer_)
        full_model.compile(optimizer='adam', loss='mse')
        return full_model

if __name__ == '__main__':
    def model():
        nn = keras.models.Sequential()
        nn.add(keras.layers.Conv2D(1, kernel_size=(1, 1), activation="relu"))
        nn.add(keras.layers.Conv2D(1, kernel_size=(1, 1), activation="relu"))
        nn.add(keras.layers.Flatten())
        return nn
        
    def processing(x, y, x_domain, y_domain):
        return x, y 

    Trainer(model(), processing)
        