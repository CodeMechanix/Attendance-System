import os
import face_recognition
import numpy as np
import cv2

# make a list of all the available images
images = os.listdir('images')
known_face_encodings=[]
known_face_names=[]

for i in os.listdir('images'):
    pic = face_recognition.load_image_file('images/'+i)
    known_face_encodings.append(face_recognition.face_encodings(pic)[0])
    known_face_names.append(i[:i.rfind('_')])

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

#cap=cv2.VideoCapture("rtsp://admin:admin@192.168.0.251:554/cam/realmonitor?channel=1@subtype=1")
cap = cv2.VideoCapture(0);
scaling_factor = 0.5

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    qray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(qray, 1.3, 5)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),3)
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    image_to_be_matched_encoded = face_recognition.face_encodings(frameRGB)
    cv2.imwrite('raw_img/1.jpg', frameRGB)
    if len(image_to_be_matched_encoded)>0:
        for (i, image) in enumerate(known_face_encodings):
            # encode the loaded image into a feature vector
            current_image_encoded = image_to_be_matched_encoded[0]
            # match your image with the image and check if it matches
            result = face_recognition.compare_faces([image], current_image_encoded)
            #print(result)
            if result[0] == True:
                distance = np.linalg.norm(image-current_image_encoded)
                #check if it was a match
                if distance<=0.40:
                    str = known_face_names[i]
#                    print(str)
#                    print(i)
                    print("Matched: " + known_face_names[i])
    # Display the resulting image
    cv2.imshow('Face Detector', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()