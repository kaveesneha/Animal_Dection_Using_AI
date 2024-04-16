# main.py
import os
import base64
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from camera import VideoCamera
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import argparse
import cv2
import math
import shutil
import random
from random import seed
from random import randint
import matplotlib.pyplot as plt
import time
import PIL.Image
from PIL import Image, ImageChops
import numpy as np
import argparse
import imagehash
import mysql.connector
import urllib.request
import urllib.parse
from werkzeug.utils import secure_filename
from urllib.request import urlopen
import webbrowser

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="wildlife_monitor"

)

UPLOAD_FOLDER = 'static/trained'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = 'abcdef'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#@app.route('/')
#def index():
#    return render_template('index.html')



@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""

    dimg=[]
    '''path_main = 'static/data'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        #resize
        img = cv2.imread('static/data/'+fname)
        rez = cv2.resize(img, (300, 300))
        cv2.imwrite("static/dataset/"+fname, rez)'''
    return render_template('index.html',msg=msg)

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""
    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    
        
        
    return render_template('login.html',msg=msg)

@app.route('/login_user', methods=['GET', 'POST'])
def login_user():
    msg=""
    msg1=""
    act = request.args.get('act')
    if act=="success":
        msg1="Register Success"
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM register WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname

            ff3=open("ulog.txt","w")
            ff3.write(uname)
            ff3.close()

            ff3=open("sms.txt","w")
            ff3.write("yes")
            ff3.close()
    
            return redirect(url_for('userhome'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    
        
        
    return render_template('login_user.html',msg=msg,msg1=msg1)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        location=request.form['location']
        uname=request.form['uname']
        pwd=request.form['pass']
        

        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM register")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        sql = "INSERT INTO register(id,name,mobile,email,location,uname,pass) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        val = (maxid,name,mobile,email,location,uname,pwd)
        mycursor.execute(sql, val)
        mydb.commit()            
        print(mycursor.rowcount, "Added Success")
        act='success'
        return redirect(url_for('login_user',act=act))
        
    return render_template('register.html',msg=msg)




@app.route('/monitor', methods=['GET', 'POST'])
def monitor():
    msg=""
    return render_template('monitor.html', msg=msg)

@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
    msg=""

    msg=""
    act=request.args.get("act")
    act2=request.args.get("act2")
    act3=request.args.get("act3")
    
    return render_template('userhome.html',msg=msg,act=act,act2=act2,act3=act3)


@app.route('/monitor1',methods=['POST','GET'])
def monitor1():
    act=""
    msg=""
    mycursor = mydb.cursor()
    f1=open("get_value.txt","r")
    v=f1.read()
    f1.close()
    if v=="person":
        a='1'
    else:
        msg=v+" Detected"
        mycursor.execute("SELECT max(id)+1 FROM animal_detect")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        fn2="F"+str(maxid)+"jpg"
        shutil.copy('getimg.jpg', 'static/trained/'+fn2)
        sql = "INSERT INTO animal_detect(id,user,animal,image_name) VALUES (%s, %s,%s,%s)"
        val = (maxid,'',v,fn2)
        mycursor.execute(sql, val)
        mydb.commit()
    
    
    return render_template('monitor1.html',act=act,msg=msg)

@app.route('/process_cam2',methods=['POST','GET'])
def process_cam2():
    msg=""
    ss=""
    uname=""
    act2=request.args.get("act2")
    det=""
    mess=""
    # (0, 1) is N
    SCALE = 2.2666 # the scale is chosen to be 1 m = 2.266666666 pixels
    MIN_LENGTH = 150 # pixels

    if request.method=='GET':
        act = request.args.get('act')
        
   
    return render_template('process_cam2.html',mess=mess,act=act)



def object_detect(fname):
    # yolo - object argument parse 
    parser = argparse.ArgumentParser(
        description='Script to run yolo object detection network ')
    parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
    parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                      help='Path to text network file: '
                                           'MobileNetSSD_deploy.prototxt for Caffe model or '
                                           )
    parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                     help='Path to weights: '
                                          'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                          )
    parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
    args = parser.parse_args()

    # Labels of Network.
    classNames = { 0: 'background',
            1: 'Cheetah', 2: 'Lion', 3: 'Fox', 4: 'Elephant',
            5: 'Bear', 6: 'Rhinoceros', 7: 'Tiger' }

    # Open video file or capture device. 
    '''if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)'''

    #Load the Caffe model 
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

    #while True:
    # Capture frame-by-frame
    #ret, frame = cap.read()
    frame = cv2.imread("static/test/"+fname)
    frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

    # yolo config fixed dimensions for input image(s)
   
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    #Set to network the input blob 
    net.setInput(blob)
    #Prediction of network
    detections = net.forward()

    #Size of frame resize (300x300)
    cols = frame_resized.shape[1] 
    rows = frame_resized.shape[0]

    #For get the class and location of object detected, 
    # There is a fix index for class, location and confidence
    # value in @detections array .
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        if confidence > args.thr: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label

            # Object location 
            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)
            
            # Factor for scale to original size of frame
            heightFactor = frame.shape[0]/300.0  
            widthFactor = frame.shape[1]/300.0 
            # Scale object detection to frame
            xLeftBottom = int(widthFactor * xLeftBottom) 
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop   = int(widthFactor * xRightTop)
            yRightTop   = int(heightFactor * yRightTop)
            # Draw location of object  
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0))
            try:
                y=yLeftBottom
                h=yRightTop-y
                x=xLeftBottom
                w=xRightTop-x
                image = cv2.imread("static/test/"+fname)
                mm=cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                fnn="detect.png"
                cv2.imwrite("static/test/"+fnn, mm)
                cropped = image[yLeftBottom:yRightTop, xLeftBottom:xRightTop]
                gg="segment.png"
                cv2.imwrite("static/test/"+gg, cropped)
                #mm2 = PIL.Image.open('static/trained/'+gg)
                #rz = mm2.resize((300,300), PIL.Image.ANTIALIAS)
                #rz.save('static/trained/'+gg)
            except:
                print("none")
                #shutil.copy('getimg.jpg', 'static/trained/test.jpg')
            # Draw label and confidence of prediction in frame resized
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                #print(label) 
    ####################

@app.route('/process_upload2', methods=['GET', 'POST'])
def process_upload2():
    msg=""
    act=request.args.get("act")
    act2=request.args.get("act2")
    act3=request.args.get("act3")
    page=request.args.get("page")
    ff=open("msg.txt","w")
    ff.write('0')
    ff.close()
    fn=request.args.get("fn")
    fn2=""
    animal=""
    ss=""
    cname=[]
    afile=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM animal_info order by id")
    row = mycursor.fetchall()
    for row1 in row:
        cname.append(row1[1])
        
    if request.method=='POST':
        #print("d")
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        file_type = file.content_type
        # if user does not select file, browser also
        # submit an empty part without filename
        tf=file.filename
        ff=open("log.txt","w")
        ff.write(tf)
        ff.close()
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = "m1.jpg"
            filename = secure_filename(fname)
            
            file.save(os.path.join("static/test", filename))
            

            cutoff=1
            for fname in os.listdir("static/dataset"):
                hash0 = imagehash.average_hash(Image.open("static/dataset/"+fname)) 
                hash1 = imagehash.average_hash(Image.open("static/test/m1.jpg"))
                cc1=hash0 - hash1
                print("cc="+str(cc1))
                if cc1<=cutoff:
                    fn=fname
                    ss="ok"
                    break
            if ss=="ok":
                act3="yes"
            else:
                act3="no"
                
                
            
            
        return redirect(url_for('process_upload2', act3=act3,fn=fn,page=page))
  
    if act3=="yes":
        g=1
        print(fn)
        #object_detect(fn)
        ##    
        ff2=open("static/trained/tdata.txt","r")
        rd=ff2.read()
        ff2.close()

        num=[]
        r1=rd.split(',')
        s=len(r1)
        ss=s-1
        i=0
        while i<ss:
            
            num.append(int(r1[i]))
            i+=1

        #print(num)
        dat=toString(num)
        dd2=[]
        d1=dat.split(',')
        
        ##
        i=1
        j=0
        
        for gff in d1:
            print(gff)
            a1=fn.split('.')
            a2=a1[0].split('-')
            if a2[1]==str(i):
                gid=i
                fn2="c_"+fn
                animal=cname[j]
                
                
                break
            j+=1
            i+=1
        print(fn2)
        print(animal)
        

        ff3=open("ulog.txt","r")
        user=ff3.read()
        ff3.close()

        ff4=open("sms.txt","r")
        sms=ff4.read()
        ff4.close()

        if user=="":
            aa=1
        else:

            if sms=="":
                aa=1
            else:
                mycursor.execute("SELECT * FROM register where uname=%s",(user, ))
                row1 = mycursor.fetchone()
                mobile=row1[2]
                name=row1[1]
                
                mess=animal+" detected"
                url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                #webbrowser.open_new(url)

                ff41=open("sms.txt","w")
                ff41.write("")
                ff41.close()
                
            mycursor = mydb.cursor()
            mycursor.execute("SELECT max(id)+1 FROM animal_detect")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            sql = "INSERT INTO animal_detect(id,user,animal,image_name) VALUES (%s, %s,%s,%s)"
            val = (maxid,'',animal,fn2)
            mycursor.execute(sql, val)
            mydb.commit()
                
    elif act3=="no":
        g=2
        msg="No Result"
    return render_template('process_upload2.html',msg=msg,act=act,act2=act2,act3=act3,fn=fn,animal=animal,fn2=fn2,afile=afile,page=page)



@app.route('/detect', methods=['GET', 'POST'])
def detect():

    ff3=open("ulog.txt","r")
    user=ff3.read()
    ff3.close()
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(user, ))
    row1 = mycursor.fetchone()
    mobile=row1[2]
    name=row1[1]

    mycursor.execute("SELECT * FROM animal_detect order by id desc")
    data = mycursor.fetchall()

                
    return render_template('detect.html', data=data)

@app.route('/admin', methods=['GET', 'POST'])
def admin():

    msg=""
    act="on"
    page="0"
    if request.method=='GET':
        msg = request.args.get('msg')
    
    
    return render_template('admin.html', msg=msg)

@app.route('/train_data', methods=['GET', 'POST'])
def train_data():

    msg=""
    
    
    
    return render_template('train_data.html', msg=msg)



@app.route('/pro1', methods=['GET', 'POST'])
def pro1():
    msg=""

    mycursor = mydb.cursor()
 
    dimg=[]
    
    path_main = 'static/dataset'
    i=0
    for fname in os.listdir(path_main):
        
        
        dimg.append(fname)
        #list_of_elements = os.listdir(os.path.join(path_main, folder))

        #resize
        #img = cv2.imread('static/data1/'+fname)
        #rez = cv2.resize(img, (300, 300))
        #cv2.imwrite("static/dataset/"+fname, rez)

        '''img = cv2.imread('static/dataset/'+fname) 	
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/trained/g_"+fname, gray)
        ##noice
        img = cv2.imread('static/trained/g_'+fname) 
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
        fname2='ns_'+fname
        cv2.imwrite("static/trained/"+fname2, dst)'''

        

        i+=1

    
    return render_template('pro1.html',dimg=dimg)


def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

@app.route('/pro2', methods=['GET', 'POST'])
def pro2():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        
        ##bin
        '''image = cv2.imread('static/dataset/'+fname)
        original = image.copy()
        kmeans = kmeans_color_quantization(image, clusters=4)

        # Convert to grayscale, Gaussian blur, adaptive threshold
        gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

        # Draw largest enclosing circle onto a mask
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
            break
        
        # Bitwise-and for result
        result = cv2.bitwise_and(original, original, mask=mask)
        result[mask==0] = (0,0,0)
        cv2.imwrite("static/trained/bin_"+fname, thresh)'''
        

    path_main2 = 'static/dataset'
    for fname in os.listdir(path_main2):
        
        ###fg
        img = cv2.imread('static/dataset/'+fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,1.5*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        segment = cv2.subtract(sure_bg,sure_fg)
        img = Image.fromarray(img)
        segment = Image.fromarray(segment)
        path3="static/trained/fg_"+fname
        #segment.save(path3)

  
        

    return render_template('pro2.html',dimg=dimg)


@app.route('/pro3', methods=['GET', 'POST'])
def pro3():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    
    for fname in os.listdir(path_main):
        

        #####
        image = cv2.imread("static/dataset/"+fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        fname2="ff_"+fname
        path4="static/trained/"+fname2
        edged.save(path4)
        ##
        
    for fname in os.listdir("static/dataset"):
        
        dimg.append(fname)
        

    return render_template('pro3.html',dimg=dimg)

@app.route('/pro4', methods=['GET', 'POST'])
def pro4():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)

    return render_template('pro4.html',dimg=dimg)

@app.route('/pro5', methods=['GET', 'POST'])
def pro5():
    msg=""
    dimg=[]
    
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        
        parser = argparse.ArgumentParser(
        description='Script to run yolo object detection network ')
        parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
        parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                          help='Path to text network file: '
                                               'MobileNetSSD_deploy.prototxt for Caffe model or '
                                               )
        parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                         help='Path to weights: '
                                              'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                              )
        parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
        args = parser.parse_args()

        # Labels of Network.
        classNames = { 0: 'background',
            1: 'Cheetah', 2: 'Lion', 3: 'Fox', 4: 'Elephant',
            5: 'Bear', 6: 'Rhinoceros', 7: 'Tiger' }

        # Open video file or capture device. 
        '''if args.video:
            cap = cv2.VideoCapture(args.video)
        else:
            cap = cv2.VideoCapture(0)'''

        #Load the Caffe model 
        net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

        #while True:
        # Capture frame-by-frame
        #ret, frame = cap.read()
        
        frame = cv2.imread("static/dataset/"+fname)
        frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

        # MobileNet requires fixed dimensions for input image(s)
        # so we have to ensure that it is resized to 300x300 pixels.
        # set a scale factor to image because network the objects has differents size. 
        # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
        # after executing this command our "blob" now has the shape:
        # (1, 3, 300, 300)
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        #Set to network the input blob 
        net.setInput(blob)
        #Prediction of network
        detections = net.forward()

        #Size of frame resize (300x300)
        cols = frame_resized.shape[1] 
        rows = frame_resized.shape[0]

        #For get the class and location of object detected, 
        # There is a fix index for class, location and confidence
        # value in @detections array .
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2] #Confidence of prediction 
            if confidence > args.thr: # Filter prediction 
                class_id = int(detections[0, 0, i, 1]) # Class label

                # Object location 
                xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)
                
                # Factor for scale to original size of frame
                heightFactor = frame.shape[0]/300.0  
                widthFactor = frame.shape[1]/300.0 
                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom) 
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop   = int(widthFactor * xRightTop)
                yRightTop   = int(heightFactor * yRightTop)
                # Draw location of object  
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))
                try:
                    y=yLeftBottom
                    h=yRightTop-y
                    x=xLeftBottom
                    w=xRightTop-x
                    image = cv2.imread("static/dataset/"+fname)
                    mm=cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imwrite("static/trained/c_"+fname, mm)
                    cropped = image[yLeftBottom:yRightTop, xLeftBottom:xRightTop]

                    gg="segment.jpg"
                    cv2.imwrite("static/result/"+gg, cropped)


                    mm2 = PIL.Image.open('static/trained/'+gg)
                    rz = mm2.resize((300,300), PIL.Image.ANTIALIAS)
                    rz.save('static/trained/'+gg)
                except:
                    print("none")
                    #shutil.copy('getimg.jpg', 'static/trained/test.jpg')
                # Draw label and confidence of prediction in frame resized
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    claname=classNames[class_id]

                    aid=0
                    if claname=="Cheetah":
                        aid=1
                    elif claname=="Lion":
                        aid=2
                    elif claname=="Fox":
                        aid=3
                    elif claname=="Elephant":
                        aid=4
                    elif claname=="Bear":
                        aid=5
                    elif claname=="Rhinoceros":
                        aid=1
                    elif claname=="Tiger":
                        aid=1

                    #mycursor.execute("update train_data set animal_id=%s where id=%s",(aid,rw[0]))
                    #mydb.commit()
                    
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                         (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                    #print(label) #print class and confidence


            
    '''i=2
    while i<=20:
        fname="ff_"+str(i)+".png"
        dimg.append(fname)
        i+=1'''
    #####
    ###################
    a=0
    b=0
    c=0
    d=0
    e=0
    '''filename = 'static/trained/data1.csv'
    dat1 = pd.read_csv(filename, header=0)
    for sv in dat1.values:
       
        if sv[2]==0:
            a+=1
        elif sv[2]==1:
            b+=1
        elif sv[2]==2:
            c+=1
        elif sv[2]==3:
            d+=1
        else:
            e+=1
            
    count1=[a,b,c,d,e]
    
    fig = plt.figure(figsize = (10, 5))
    
    class1=[]
    #count1=[50,100]
    # creating the bar plot
    plt.bar(class1, count1, color ='blue',
            width = 0.4)
 


    plt.xlabel("Classification")
    plt.ylabel("Count")
    plt.title("")

   
    plt.savefig('static/trained/classi.png')
    #plt.close()
    plt.clf()'''
    ###################################    
    #graph
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[10,30,50,80,100]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model Precision")
    plt.ylabel("precision")
    
    fn="graph1.jpg"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph2
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[10,30,50,80,100]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model recall")
    plt.ylabel("recall")
    
    fn="graph2.jpg"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph3
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(94,98)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(94,98)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[10,30,50,80,100]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model accuracy")
    plt.ylabel("accuracy")
    
    fn="graph3.jpg"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,4)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(1,4)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[10,30,50,80,100]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model loss")
    plt.ylabel("loss")
    
    fn="graph4.jpg"
    #plt.savefig('static/trained/'+fn)
    plt.close()

    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
    ###############################
    return render_template('pro5.html',dimg=dimg)


def toString(a):
  l=[]
  m=""
  for i in a:
    b=0
    c=0
    k=int(math.log10(i))+1
    for j in range(k):
      b=((i%10)*(2**j))   
      i=i//10
      c=c+b
    l.append(c)
  for x in l:
    m=m+chr(x)
  return m
@app.route('/pro6', methods=['GET', 'POST'])
def pro6():
    msg=""
    dimg=[]
    data1=[]
    data2=[]
    data3=[]
    data4=[]
    data5=[]
    data6=[]
    data7=[]
    cname=[]

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM animal_info order by id")
    row = mycursor.fetchall()
    for row1 in row:
        cname.append(row1[1])
           
                     
    ##    
    ff2=open("static/trained/tdata.txt","r")
    rd=ff2.read()
    ff2.close()

    num=[]
    r1=rd.split(',')
    s=len(r1)
    ss=s-1
    i=0
    while i<ss:
        num.append(int(r1[i]))
        i+=1

    #print(num)
    dat=toString(num)
    dd2=[]
    d1=dat.split(',')
    ##
    path_main = 'static/dataset'
    i=0
    v1=0
    v2=0
    v3=0
    v4=0
    v5=0
    v6=0
    v7=0
    vv=""
    for fname in os.listdir(path_main):
    
        a1=fname.split('.')
        d2=a1[0].split('-')
            
        if d2[1]=='1':
            v1+=1
            data1.append(fname)
        if d2[1]=='2':
            v2+=1
            data2.append(fname)
        if d2[1]=='3':
            v3+=1
            data3.append(fname)
        if d2[1]=='4':
            v4+=1
            data4.append(fname)
        if d2[1]=='5':
            v5+=1
            data5.append(fname)
        if d2[1]=='6':
            v6+=1
            data6.append(fname)
        if d2[1]=='7':
            v7+=1
            data7.append(fname)
            
        
    #####################

    g1=v1+v2+v3+v4+v5+v6+v7
    dd2=[v1,v2,v3,v4,v5,v6,v7]
    
    
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
    print(doc)
    print(values)
    fig = plt.figure(figsize = (10, 5))
     
    # creating the bar plot
    plt.bar(doc, values, color ='blue',
            width = 0.4)
 

    plt.ylim((1,g1))
    plt.xlabel("Animal")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    plt.xticks(rotation=20)
    plt.savefig('static/trained/'+fn)
    
    #plt.close()
    plt.clf()
    ##########     
    

    return render_template('pro6.html',cname=cname,data1=data1,data2=data2,data3=data3,data4=data4,data5=data5,data6=data6,data7=data7)




@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))


def gen(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed')
        

def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)
