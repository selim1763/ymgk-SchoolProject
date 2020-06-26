
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


veri = pd.read_csv('model_dataset.csv')

#Sınıf sayısı ve etiketlerin belirlenmesi
label_encoder = LabelEncoder().fit(veri.Class)
labes = label_encoder.transform(veri.Class)
classes = list(label_encoder.classes_)

#Girdi ve çıktı verilerinin hazırlanması
x = veri.drop(["Class"],axis=1)
y=labes
nb_features = 1
nb_classes = len(classes)

#Verilerin standartlaşması
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X = sc.fit_transform(x)
X = x

#Eğitim ve test verilerinin hazırlamanması
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test =train_test_split(X,y,test_size = 0.3)


# çıktı değerlerinin kategorileştirmesi
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#ÈDerin öğrenme yapabilmesi için 3 boyutlu hale getirmeliyiz
X_train = np.array(X_train).reshape(84,1,1)
X_test = np.array(X_test).reshape(36,1,1)


# RNN modelinin oluşturulması
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Flatten, BatchNormalization

model = Sequential()

model.add(LSTM(512, input_shape=(nb_features,1)))

model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.15))
model.add(Dense(2048, activation = "relu"))
model.add(Dense(1024, activation = "relu"))
model.add(Dense(nb_classes, activation ="softmax"))
model.summary()

#♦Modelin derlenmesi
from tensorflow.keras.optimizers import SGD
opt = SGD(lr=1e-3, decay = 1e-5,momentum = 0.9,nesterov =True)
model.compile (loss="categorical_crossentropy", optimizer='adam', metrics =["accuracy"])

#modeliin Eğitilmesi
print(X_train)
print(y_train)

score = model.fit(X_train,y_train,epochs=50, validation_data=(X_test,y_test))
#_, acc = model.evaluate(X_train, y_train, verbose=0)


def get_predict(input):
    pr = np.array([[
        input
              ]])
    result = model.predict(pr)
    return result


import asyncio
import struct
import ctypes

class Client:
    def __init__(self,transport):
        self.transport = transport

    def send(self,data: bytearray):
        if(self.transport != None):
            self.transport.write(data)


class ServerProtocol(asyncio.Protocol):

    def connection_made(self, transport):
        peername = transport.get_extra_info('peername')
        print('Connection from {}'.format(peername))

        self.client = Client(transport)

    def data_received(self, data: bytes):
        buffer = bytearray(data)
        length = len(data)

        value = struct.unpack_from('b', buffer, 0)
        result = get_predict(value)

        struct.pack_into('b', buffer, 0, result)
        self.client.send(buffer)


    def connection_lost(self, exc):
        pass


loop = asyncio.get_event_loop()
coro = loop.create_server(ServerProtocol, '127.0.0.1',8888)
server = loop.run_until_complete(coro)

print('Server started on {}'.format(server.sockets[0].getsockname() ))

try:
    loop.run_forever()
except KeyboardInterrupt:
    pass

server.close()
loop.run_until_complete(server.wait_closed())
loop.close()






"""
#Gerekli bilgilerin verilmesi
print("Ortalama Başarım",np.mean(model.history.history["val_accuracy"]))
print("Ortalama Kayıp",np.mean(model.history.history["val_loss"]))



#Sonuçları grafiğer dökme
import matplotlib.pyplot as plt
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Başarımı")
plt.xlabel("Epok")
plt.ylabel("Başarım")
plt.legend(["Eğitim","Test"], loc="upper_left")
plt.show()

#Kayıpların gösterilmesi
import matplotlib.pyplot as plt
plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Kayıpları")
plt.xlabel("Epok")
plt.ylabel("Kayıp")
plt.legend(["Eğitim","Test"], loc="upper_left")
plt.show()

"""





