import cv2
import numpy as np
import face_recognition as faceRegLib

demo_img_bgr = faceRegLib.load_image_file('demo_tutorial_images/sample.jpg')
emo_img_rgb = cv2.cvtColor(demo_img_bgr,cv2.COLOR_BGR2RGB)
cv2.imshow('bgr', img_bgr)
cv2.imshow('rgb', img_rgb)
cv2.waitKey

original_img=faceRegLib.load_image_file('demo_tutorial_images/obama.jpg')
original_img_rgb = cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB)

face = face_recognition.face_locations(obama_img_rgb)[0] 
copy = original_img_rgb.copy()

cv2.rectangle(copy, (face[3], face[0]),(face[1], face[2]), (255,0,255), 2)
cv2.imshow('copy', copy)
cv2.imshow('Original',original_img_rgb)
cv2.waitKey(0)

img_sample = faceRegLib.load_image_file('demo_images/sample_image.jpg')
img_sample1 = cv2.cvtColor(img_sample,cv2.COLOR_BGR2RGB)
original_face = faceRegLib.face_locations(img_sample)[0] 
demo_train_encode = faceRegLib.face_encodings(img_sample)[0]
demo = faceRegLib.load_image_file('demo_image/sample_image.jpg')
demo = cv2.cvtColor(demo, cv2.COLOR_BGR2RGB)
demo_encode = faceRegLib.face_encodings(demo)[0] 
print(faceRegLib.compare_faces([demo_train_encode],demo_encode))
cv2.rectangle(img_obama, (face[3], face[0]),(face[1], face[2]), (255,0,255), 1)
cv2.imshow('OBAMA', img_sample)
cv2.waitKey(0)