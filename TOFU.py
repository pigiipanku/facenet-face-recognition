import os
import glob
import datetime
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from keras import backend as K
from fr_utils import *
from inception_blocks_v2 import *



#model
K.set_image_data_format('channels_first')
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def triplet_loss(y_true, y_pred, alpha=0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,
                                                   positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,
                                                   negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss

FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
load_weights_from_FaceNet(FRmodel)

#database   want to make 1&2
def prepare_database():
    database = {}
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, FRmodel)
    return database

#write
#database = prepare_database()
#with open('database.binaryfile','wb') as f:
#   pickle.dump(database,f)
#read
f = open('database.binaryfile','rb')
database = pickle.load(f)

#TOFU
def judge(image,database,model):
    encoding = img_to_encoding(image,model)

    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' % (name, dist))
        if dist < min_dist:
            min_dist = dist
            identity = name

    return identity,min_dist

#judgement
cap = cv2.VideoCapture(0)
HAAR_FILE = 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(HAAR_FILE)
while True:
    ret,frame = cap.read()
    cv2.imshow("frame",frame)
    if not ret:
        print('error')
        continue
    if cv2.waitKey(1) & 0xFF == ord('p'):
        cv2.imwrite('frame.jpg',frame)
        image = cv2.imread('frame.jpg')
        image_gray = cv2.imread('frame.jpg', 0)
        face = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=3, minSize=(240, 240))
        if len(face) > 0:
            for x,y,w,h in face:
                face_cut = image[y:y+h, x:x+w]
            if not os.path.exists('storage'):
                os.mkdir('storage')
            now = datetime.datetime.now()
            d = datetime.datetime(now.year,now.month,now.day,now.hour,now.minute,now.second)
            cv2.imwrite(f'storage/{d}.jpg',face_cut)
            print('--success--')

            identity,distance = judge(face_cut,database,FRmodel)
            if distance < 0.52:
                print('a')
                identity = identity + '. Your identity is my school student!!'

            plt.imshow(face_cut)
            plt.title(identity + '{:.2f}'.format(float(100 - 2 * (distance-0.52) * 100)) + '%')
            plt.xticks([]),plt.yticks([])
            plt.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sys.exit()
print('---END---')