import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
import PIL.ImageOps
import os,ssl,time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X,y=fetch_openml("mnist_784", version=1, return_X_y=True)
print(pd.Series(y).value_counts())
classes=["0","1","2","3","4","5","6","7","8","9"]
nclasses=len(classes)

xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
xtrainscale=xtrain/255.0
xtestscale=xtest/255.0

clf=LogisticRegression(solver="saga",multi_class="multinomial").fit(xtrainscale,ytrain)

ypred=clf.predict(xtestscale)
accuracy=accuracy_score(ytest,ypred)
print(accuracy)

cap =cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=gray.shape
        upperLeft=(int(width/2-56),int(height/2-56))
        bottomRight=(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upperLeft,bottomRight,(0,255,0),2)
        roi=gray[upperLeft[1]:bottomRight[1],upperLeft[0]:bottomRight[0]]
        im_pil=Image.fromarray(roi)
        Imagebw=im_pil.convert("L")
        Imagebw_resize=Imagebw.resize((28,28),Image.ANTIALIAS)
        Imagebw_resize_inverted=PIL.ImageOps.invert(Imagebw_resize)
        pixelfilter=20
        minpixel=np.percentile(Imagebw_resize_inverted,pixelfilter)
        Imagebw_resize_inverted_scaled=np.clip(Imagebw_resize_inverted-minpixel,0,255)
        maxpixel=np.max(Imagebw_resize_inverted) 
        Imagebw_resize_inverted_scaled=np.asarray(Imagebw_resize_inverted_scaled)/maxpixel
        testsample=np.array(Imagebw_resize_inverted_scaled).reshape(1,784)
        testpred=clf.predict(testsample)
        print("predicted class is:", testpred)

        cv2.imshow("frame",gray)
        if cv2.waitKey(1) & 0xFF==ord("q"):
            break
    
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()

