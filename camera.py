# camera.py

import cv2
import PIL.Image
from PIL import Image
import argparse
import shutil
class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        
        self.video = cv2.VideoCapture(0)
        self.k=1
        #cap = self.video
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        #self.video = cv2.VideoCapture('video.mp4')

        # Check if camera opened successfully
        #if (cap.isOpened() == False): 
        #  print("Unable to read camera feed")

        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        #frame_width = int(cap.get(3))
        #frame_height = int(cap.get(4))

        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        #self.out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


        
    
    def __del__(self):
        self.video.release()
        
    
    def get_frame(self):
        success, image = self.video.read()
        #self.out.write(image)

        cv2.imwrite("getimg.jpg", image)
        
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Read the frame
        #_, img = cap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw the rectangle around each face
        j = 1

        '''ff=open("user.txt","r")
        uu=ff.read()
        ff.close()

        ff1=open("photo.txt","r")
        uu1=ff1.read()
        ff1.close()'''
        
        
        ###########################################
        # construct the argument parse 
        parser = argparse.ArgumentParser(
            description='Script to run MobileNet-SSD object detection network ')
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
            1: 'mobile', 2: 'bicycle', 3: 'cup', 4: 'glass',
            5: 'bottle', 6: 'paper', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
            14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'elephant', 18: 'sofa', 19: 'cell phone', 20: 'tvmonitor' }

        # Open video file or capture device. #plastic
        '''if args.video:
            cap = cv2.VideoCapture(args.video)
        else:
            cap = cv2.VideoCapture(0)'''

        #Load the Caffe model 
        net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

        #while True:
        # Capture frame-by-frame
        #ret, frame = cap.read()
        frame_resized = cv2.resize(image,(300,300)) # resize frame for prediction

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
                heightFactor = image.shape[0]/300.0  
                widthFactor = image.shape[1]/300.0 
                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom) 
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop   = int(widthFactor * xRightTop)
                yRightTop   = int(heightFactor * yRightTop)
                # Draw location of object  
                cv2.rectangle(image, (xRightTop, yRightTop), (xLeftBottom, yLeftBottom),
                              (0, 255, 0),2)
                #print("x="+str(xRightTop)+" x+w="+str(xLeftBottom))
                #print("y="+str(yRightTop)+" y+h="+str(yLeftBottom))
                try:
                    
                    image = cv2.imread("getimg.jpg")
                    cropped = image[yLeftBottom:yRightTop, xLeftBottom:xRightTop]
                    gg="test.jpg"
                    cv2.imwrite("static/trained/"+gg, cropped)
                    mm2 = PIL.Image.open('static/trained/'+gg)
                    rz = mm2.resize((300,300), PIL.Image.ANTIALIAS)
                    rz.save('static/trained/'+gg)
                except:
                    shutil.copy('getimg.jpg', 'static/trained/test.jpg')
                label=""
                # Draw label and confidence of prediction in frame resized
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    claname=classNames[class_id]
                    
                    if claname=="":
                        s=1
                    else:
                        ff1=open("get_value.txt","w")
                        ff1.write(claname)
                        ff1.close()
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(image, (xLeftBottom, yLeftBottom - labelSize[1]),
                                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                         (255, 255, 255), cv2.FILLED)
                    cv2.putText(image, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        ###########################################
        
        '''for (x, y, w, h) in faces:
            mm=cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imwrite("static/myface.jpg", mm)

            
            image = cv2.imread("static/myface.jpg")
            cropped = image[y:y+h, x:x+w]
            gg="f"+str(j)+".jpg"
            cv2.imwrite("static/faces/"+gg, cropped)

            ###
            if self.k<=40:
                self.k+=1
                fnn=""
                fnn=uu+"_"+str(self.k)+".jpg"
                

                ff2=open("det.txt","w")
                ff2.write(str(self.k))
                ff2.close()
                if uu1=="2":
                    cv2.imwrite("static/frame/"+fnn, cropped)
                
                mm2 = PIL.Image.open('static/faces/'+gg)
                rz = mm2.resize((100,100), PIL.Image.ANTIALIAS)
                rz.save('static/faces/'+gg)
                #rz.save("upload/"+gg)
            j += 1

        ff4=open("img.txt","w")
        ff4.write(str(j))
        ff4.close()   ''' 

            

        

            
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
