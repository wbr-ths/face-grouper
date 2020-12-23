import face_recognition
import os
import cv2
from os import listdir
from matplotlib import pyplot as plt
import pickle
import numpy as np
from operator import add
import time

from utils import improve_contrast_image_using_clahe, image_resize, image_size_limit

IMAGE_PATH = 'images/test'
ENCODINGS = 'encodings.dat'
TOLERANCE = 0.55 # 0.51
MODEL = 'cnn'


def load_face_encodings(image_path, encodings_path):

    face_id = 0 # keeps track of person-id

    # checking for cache
    if encodings_path in os.listdir():
        print('load encodings from cache')
        with open(encodings_path, 'rb') as f:
            return pickle.load(f)
    else:
        print('load encodings from files')
        encodings = []
        crop_sizes = []
        files = os.listdir(image_path)
        max_len = len(max(files, key=len))

        for file in files:

            print('loading', file, end=" "*max_len + '\r', flush=True)

            # load image-file
            image = face_recognition.load_image_file(f'{image_path}/{file}')

            # preprocess image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = improve_contrast_image_using_clahe(image)
            image = image_size_limit(image, 1920, 1080)

            # find faces in image
            locations = face_recognition.face_locations(image, model=MODEL, number_of_times_to_upsample=1)
            encoding = face_recognition.face_encodings(image, locations)#, num_jitters=100)
            encodings += encoding

            for face_location in locations:

                # crop face out of the image
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])
                crop = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                # get size of the face
                h, w, _ = crop.shape
                crop_size = h*w
                crop_sizes.append(crop_size)

                # save face in /crops
                crop = image_resize(crop, height=256)
                cv2.imwrite(f'crops/{face_id}.png', crop)
                face_id += 1

                '''
                color = [0, 0, 255]
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])
                image = cv2.rectangle(image, top_left, bottom_right, color, 2)
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(image, str(locations.index(face_location)), (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                '''

            '''winname = f'{image_path}/{file}'
            cv2.imshow(winname, image)
            cv2.moveWindow(winname, 0,0)
            cv2.waitKey(0)
            cv2.destroyWindow(winname)'''

        # sort faces by image-size
        #indexes = []
        #for i in range(len(encodings)):
        #    indexes.append(i)
        #print(indexes)

        temp = []
        for i in range(len(encodings)):
            temp.append([crop_sizes[i], encodings[i], i])
        temp = sorted(temp, key=lambda l:l[0], reverse=True)
        encs = []
        indexes = []
        for i in range(len(temp)):
            encs.append(temp[i][1])
            indexes.append(temp[i][2])

        print('all images loaded')
        with open(encodings_path, 'wb') as f:
            pickle.dump(encodings, f)
        return encs, indexes

def average_encoding(encodings):
    ret = encodings[0]
    #print(encodings)
    for i in range(1, len(encodings)):
        for j in range(len(encodings[i])):
            ret[j] += encodings[i][j]
    for i in range(len(ret)):
        ret[i] = ret[i]/len(encodings)
    #print(ret)
    return ret

def average_encodings(encodings):
    ret = []
    for encoding in encodings:
        if len(encoding) > 1:
            tmp = average_encoding(encoding)
            ret.append(tmp)
        else:
            ret.append(encoding[0])
    return np.array(ret).tolist()


if __name__ == '__main__':

    encodings, indexes = load_face_encodings(IMAGE_PATH, ENCODINGS)
    print('found', len(encodings), 'faces')

    known = [[encodings[0]]]
    known_orgs = [[indexes[0]]]

    for i in range(1, len(encodings)):
        #print('i', i)

        #print(len(average_encodings(known)))
        #print(encodings[i])

        tmp = face_recognition.compare_faces(average_encodings(known), encodings[i], TOLERANCE)
        if True in tmp:
            index = tmp.index(True)
            #print('person found', index)
            known[index].append(encodings[i])
            known_orgs[index].append(indexes[i])
            #exit()
            #print(known[8])
        else:
            #print('new person')
            known.append([])
            known[-1].append(encodings[i])
            known_orgs.append([])
            known_orgs[-1].append(indexes[i])

    for i in range(len(known)):
        if len(known[i]) >= 1:
            for j in range(len(known[i])):
                #plt.plot(known[i][j])
            
                image = cv2.imread('crops/' + str(known_orgs[i][j]) + '.png')
                #image = cv2.imread('crops/' + str(indexes[known_orgs[i][j]]) + '.png')
                try:
                    os.mkdir('persons/' + str(i))
                except:
                    pass
                cv2.imwrite('persons/' + str(i) + '/' + str(j) + '.png', image)


                #winname = str(known_orgs[i][j])
                #cv2.imshow(winname, image)
                #cv2.moveWindow(winname, 0,0)
                #cv2.waitKey(0)
                #cv2.destroyWindow(winname)

            #plt.show()
    