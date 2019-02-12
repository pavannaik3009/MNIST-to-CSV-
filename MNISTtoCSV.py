def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
"mnist_train.csv", 60000)


import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from sklearn.naive_bayes import BernoulliNB

data=pd.read_csv("mnist_train.csv").as_matrix()
clf = BernoulliNB()

#training dateset 
xtrain=data[0:50000,1:]
train_label=data[0:50000,0]
clf.fit(xtrain,train_label)

#testing data
xtest=data[50000:,1:]
actual_label=data[50000:,0]

d=xtest[3]
d.shape=(28,28)
pt.imshow(d,cmap='gray')
print(clf.predict([xtest[3]]))
pt.show()

p=clf.predict(xtest)