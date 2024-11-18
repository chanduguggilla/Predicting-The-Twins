from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import simpledialog
from tkinter import filedialog
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import pandas as pd

main = tkinter.Tk()
main.title("Prediction of Identical Twins using ML") #designing main screen
main.geometry("1300x1200")

global filename
global X, Y
global X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore, labels, rf
global scaler
labels = ['Real', 'Twins']

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index     

def uploadDataset():
    global filename
    global X, Y
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n")
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    bilateral_filter = cv2.bilateralFilter(img,15,80,80)
                    bilateral_filter = cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize = (8, 8))
                    bilateral_filter = clahe.apply(bilateral_filter)
                    detected_image = cv2.Canny(bilateral_filter,50,150)
                    img = cv2.resize(detected_image, (32, 32))
                    X.append(img.ravel())
                    label = getID(name)
                    Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)                    
    text.insert(END,"Labels in Dataset : "+str(labels)+"\n")
    text.insert(END,"Total Real & Twins Images found in dataset : "+str(X.shape[0]))

def DatasetPreprocessing():
    text.delete('1.0', END)
    global X, Y
    global X_train, X_test, y_train, y_test, scaler
    X = X.astype('float32')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5) #split dataset into train and test
    text.insert(END,"Dataset Normalization & Preprocessing Task Completed\n\n")
    text.insert(END,"Dataset Train & Test Splits\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"80% dataset used for training  : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset user for testing   : "+str(X_test.shape[0])+"\n")


def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show() 
    
def runNaiveBayes():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore, cnn_model
    global X_train, y_train, X_test, y_test
    accuracy = []
    precision = []
    recall = [] 
    fscore = []

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    predict = nb.predict(X_test)
    calculateMetrics("Naive Bayes", y_test, predict)

def runRandomForest():
    global rf
    global X_train, y_train, X_test, y_test
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test)
    calculateMetrics("Random Forest", y_test, predict)

def graph():
    df = pd.DataFrame([['Naive Bayes','Accuracy',accuracy[0]],['Naive Bayes','Precision',precision[0]],['Naive Bayes','Recall',recall[0]],['Naive Bayes','FSCORE',fscore[0]],
                       ['Random Forest','Accuracy',accuracy[1]],['Random Forest','Precision',precision[1]],['Random Forest','Recall',recall[1]],['Random Forest','FSCORE',fscore[1]],
                      ],columns=['Algorithms','Accuracy','Value'])
    df.pivot("Algorithms", "Accuracy", "Value").plot(kind='bar')
    plt.title("All Algorithm Comparison Graph")
    plt.show() 
   

def predict():
    global rf, scaler
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    bilateral_filter = cv2.bilateralFilter(img,15,80,80)
    bilateral_filter = cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize = (8, 8))
    bilateral_filter = clahe.apply(bilateral_filter)
    detected_image = cv2.Canny(bilateral_filter,50,150)
    image = cv2.resize(detected_image, (32, 32))
    X = []
    X.append(image.ravel())
    X = np.asarray(X)
    X = X.astype('float32')
    X = scaler.transform(X)
    predict = rf.predict(X)[0]

    img = cv2.imread(filename)
    img = cv2.resize(img, (700,400))
    cv2.putText(img, 'Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    cv2.imshow('Predicted As : '+labels[predict], img)
    cv2.imshow("Detected Object", detected_image)
    cv2.waitKey(0)
    


font = ('times', 16, 'bold')
title = Label(main, text='Prediction of Identical Twins using ML')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Twins Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

preButton = Button(main, text="Dataset Preprocessing", command=DatasetPreprocessing)
preButton.place(x=300,y=100)
preButton.config(font=font1) 

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=runNaiveBayes)
nbButton.place(x=510,y=100)
nbButton.config(font=font1) 

rfButton = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
rfButton.place(x=740,y=100)
rfButton.config(font=font1) 

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=150)
graphButton.config(font=font1)

predictButton = Button(main, text="Twins or Real Face Prediction", command=predict)
predictButton.place(x=300,y=150)
predictButton.config(font=font1)  

#main.config(bg='OliveDrab2')
main.mainloop()