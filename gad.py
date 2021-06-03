import cv2
import math
import argparse

import tkinter as tk
import tkinter.filedialog as fd
from tkinter import *
from PIL import Image, ImageTk
from tkmacosx import Button
from tkinter.font import Font
import numpy as np


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            frameOpencvDnn = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2),(185, 185, 250), int(round(frameHeight/150)), 8)
            #(118, 84, 214)  #(247, 246, 198)--cyan
    
    return frameOpencvDnn,faceBoxes


def model(filepath):

    #frame = cv2.imread(filepath)
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    global frame
    frame = Image.open(filepath)
    frame = frame.resize((600,600))
    frame = np.array(frame)
    
    faceProto="../GAD-final/ProtobufAndCaffe/opencv_face_detector.pbtxt"
    faceModel="../GAD-final/ProtobufAndCaffe/opencv_face_detector_uint8.pb"
    ageProto="../GAD-final/ProtobufAndCaffe/age_deploy.prototxt"
    ageModel="../GAD-final/ProtobufAndCaffe/age_net.caffemodel"
    genderProto="../GAD-final/ProtobufAndCaffe/gender_deploy.prototxt"
    genderModel="../GAD-final/ProtobufAndCaffe/gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)


    padding=20
    
    global faceBox,resultImg,gender,age
    
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")
        label1.configure(foreground = "#fab9b9", text = "No Face Detected")

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                    min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                    :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        #print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        #print(f'Age: {age[1:-1]} years')
        
        print("Model has Successfully Run ")
    
    
        #cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX,0.7, (107, 69, 60), 2, cv2.LINE_AA)
        #cv2.imshow("Detecting age and gender", resultImg)
    

    return gender, age

    #cv2.destroyAllWindows()
    #model.save("gendAgeClassifier.h5")

top = tk.Tk()
top.geometry("1100x600")
top.title("Age Gender Detector")
top.configure(background="#333333")

label1 = tk.Label(top, background = "#333333", font = ("montserrat",15,"bold"))
label2 = tk.Label(top, background = "#333333", font = ("montserrat",15,"bold"))
pic = tk.Label(top)

filepath="../GAD-final/Data/"



def classify(filepath):
    
    print(f'Gender: {gender}')
    print(f'Age: {age[1:-1]} years')
    print("Classified")
    
    
    cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX,0.7,(185, 185, 250), 1, cv2.LINE_AA)
    cv2.imshow("Detecting age and gender", resultImg)
        
    label1.configure(foreground = "#fab9b9", text = f'Gender: {gender}')
    label2.configure(foreground = "#fab9b9", text = f'Age: {age[1:-1]} years')

    
def upload():
    try:
        global filepath
        filepath = filedialog.askopenfilename()
        print("Filepath:",filepath)
        upload = Image.open(filepath)
        upload = upload.resize((300, 300), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(upload)
        
        pic.configure(image=img)
        pic.image=img
        pic.pack(side=TOP,expand=True)
        
        print("Picture Uploaded")
        #label.configure(text="",fg="red",font=('montserrat',20, "bold"))
        #classify_button(filepath)
    
    except:
        pass

def clear_label():
    label1.config(text = "")
    label2.config(text = "")
    
    
heading = Label(top, text="Age Gender Recognition",pady=20, font=('montserrat',30))
heading.configure(background='#333333',foreground='#d9fffa')#feffd9
heading.pack(side=TOP)

clear = Button(top, text="Clear Labels", command=clear_label,borderless=1, padx=15,pady=10)
clear.configure(background="#fab9b9", foreground="#3b3c40",font=("montserrat",15))
clear.pack(side=BOTTOM,pady=50)

upload=Button(top,text="Upload Image",command=upload, borderless=1, padx=15,pady=10)
upload.configure(background='#c7fcf5', foreground='#3b3c40',font=('montserrat',20))
upload.pack(side=BOTTOM,pady= 5)

detectButton =Button(top,text="Detect Image", command = lambda : classify(filepath), borderless=1, padx=15,pady=10)
detectButton.configure(background="#d9fffa", foreground="#3b3c40",font=("montserrat",15))
detectButton.pack(side=RIGHT,pady=50,padx=10)

runModelButton =Button(top,text="Run Model", command = lambda : model(filepath), borderless=1, padx=15,pady=10)
runModelButton.configure(background="#d9fffa", foreground="#3b3c40",font=("montserrat",15))
runModelButton.pack(side=LEFT,pady=50,padx=10)

label1.pack(side=TOP,expand=True)
label2.pack(side=TOP,expand=True,pady= 5)


top.mainloop()
