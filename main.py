import face_recognition
import os
import cv2
from os import listdir
from matplotlib import pyplot as plt
import pickle
import numpy as np
from operator import add
import time
import json
from shutil import copyfile

from utils import improve_contrast_image_using_clahe, image_resize, image_size_limit, generate_weights

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
            tmp = pickle.load(f)
            return tmp[0], tmp[1]
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
            #image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
            image = improve_contrast_image_using_clahe(image)
            image = image_size_limit(image, 2200) # change if out of memory

            # find faces in image
            locations = face_recognition.face_locations(image, model=MODEL, number_of_times_to_upsample=1)
            encoding = face_recognition.face_encodings(image, locations, num_jitters=10, model='large')
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
            pickle.dump([encs, indexes], f)
        return encs, indexes

def average_encoding(encodings):
    ret = [0]*128
    weights = generate_weights(len(encodings))
    for i in range(len(encodings)):
        for j in range(len(encodings[i])):
            ret[j] += encodings[i][j]*weights[i]
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

        tmp = face_recognition.face_distance(average_encodings(known), encodings[i])
        min_dist = tmp.min()


        if min_dist <= TOLERANCE:
            #index = tmp.index(True)
            index = np.where(tmp==min_dist)[0][0]
            #print(index)
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


    nodes = [{'name': str(i)} for i in range(len(known))]
    for i in range(len(known_orgs)):
        copyfile('crops/' + str(known_orgs[i][0]) + '.png', 'static/images/' + str(i) + '.png')

   
    links = []
    for i in range(len(known)-1):
        for j in range(i+1, len(known)):
            links.append({'source': i,
                          'target': j,
                          'distance': face_recognition.face_distance([average_encoding(known[i])], known[j][0])[0]})

    data = {'nodes': nodes, 'links': links}

    with open('data.json', 'w') as f:
        json.dump(data, f)

    #for i in range(len(known)):
        #if len(known[i]) >= 1:
            #for j in range(len(known[i])):

                #pass

                #plt.plot(known[i][j])
            
                #image = cv2.imread('crops/' + str(known_orgs[i][j]) + '.png')
                #image = cv2.imread('crops/' + str(indexes[known_orgs[i][j]]) + '.png')
                #try:
                #    os.mkdir('persons/' + str(i))
                #except:
                #    pass
                #cv2.imwrite('persons/' + str(i) + '/' + str(j) + '.png', image)
                #cv2.imwrite('persons/' + str(i) + '_' + str(j) + '.png', image)


                #winname = str(known_orgs[i][j])
                #cv2.imshow(winname, image)
                #cv2.moveWindow(winname, 0,0)
                #cv2.waitKey(0)
                #cv2.destroyWindow(winname)

            #plt.show()