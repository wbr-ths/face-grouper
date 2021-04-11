import face_recognition
import os
import cv2
from os import listdir, remove, mkdir
import pickle
import numpy as np
from operator import add
import json
from shutil import copyfile, rmtree
from utils import improve_contrast_image_using_clahe, image_resize, image_size_limit, generate_weights
from app import start_app


SRC_IMAGE_PATH = 'images/'
DST_IMAGE_PATH = 'static/images/'
ENCODINGS = 'encodings.dat' # just a caching file
TOLERANCE = 0.6 # 0.6 is default
MODEL = 'cnn' # use 'hog' if you dont have CUDA


def load_face_encodings(image_path, encodings_path):
    """Load face encodings from cache or from files."""

    face_id = 0

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
            image = improve_contrast_image_using_clahe(image)
            image = image_size_limit(image, 2000) # change if out of memory

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
    """Return average encoding from a list of encoding."""
    ret = [0]*128
    weights = generate_weights(len(encodings))
    for i in range(len(encodings)):
        for j in range(len(encodings[i])):
            ret[j] += encodings[i][j]*weights[i]
    return ret


def average_encodings(encodings):
    """Return a list of average encodings from a 2d-list of encodings."""
    ret = []
    for encoding in encodings:
        if len(encoding) > 1:
            tmp = average_encoding(encoding)
            ret.append(tmp)
        else:
            ret.append(encoding[0])
    return np.array(ret).tolist()


def face_grouping(encodings, indexes):
    """Face grouping algo."""
    known = [[encodings[0]]]
    known_orgs = [[indexes[0]]]
    for i in range(1, len(encodings)):
        tmp = face_recognition.face_distance(average_encodings(known), encodings[i])
        min_dist = tmp.min()
        if min_dist <= TOLERANCE:
            index = np.where(tmp==min_dist)[0][0]
            known[index].append(encodings[i])
            known_orgs[index].append(indexes[i])
        else:
            known.append([])
            known[-1].append(encodings[i])
            known_orgs.append([])
            known_orgs[-1].append(indexes[i])
    return known, known_orgs


def generate_json(known, known_orgs, DST_IMAGE_PATH):
    """Generate data.json."""
    nodes = [{'name': str(i)} for i in range(len(known))]
    for i in range(len(known_orgs)):
        copyfile('crops/' + str(known_orgs[i][0]) + '.png', DST_IMAGE_PATH + str(i) + '.png')
    links = []
    for i in range(len(known)-1):
        for j in range(i+1, len(known)):
            links.append({'source': i,
                          'target': j,
                          'distance': face_recognition.face_distance([average_encoding(known[i])], known[j][0])[0]})
    data = {'nodes': nodes, 'links': links}
    with open('data.json', 'w') as f:
        json.dump(data, f)


def drop_cache():
    """Drop all cached files."""
    rmtree('crops')
    mkdir('crops')
    rmtree('static/images')
    mkdir('static/images')
    if 'encodings.dat' in listdir():
        remove('encodings.dat')
    if 'data.json' in listdir():
        remove('data.json')



if __name__ == '__main__':

    # drop cache
    if input('Do you want to delete the cache? (YES/no)').lower() in ['yes', 'y', '']:
        drop_cache()

    # load encodings
    encodings, indexes = load_face_encodings(SRC_IMAGE_PATH, ENCODINGS)
    print('found', len(encodings), 'faces')

    # group faces
    known, known_orgs = face_grouping(encodings, indexes)

    # generating JSON
    generate_json(known, known_orgs, DST_IMAGE_PATH)

    start_app()
