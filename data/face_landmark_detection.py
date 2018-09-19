import os
import glob
from skimage import io
import numpy as np
import dlib
import sys

if len(sys.argv) < 2 or (sys.argv[1] != 'train' and sys.argv[1] != 'test'):
    raise ValueError('usage: python data/face_landmark_detection.py [train|test]')

phase = sys.argv[1]
dataset_path = 'datasets/face/'
faces_folder_path = os.path.join(dataset_path, phase + '_img/')
predictor_path = os.path.join(dataset_path, 'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

img_paths = sorted(glob.glob(faces_folder_path + '*'))
for i in range(len(img_paths)):
    f = img_paths[i]
    print("Processing video: {}".format(f))
    save_path = os.path.join(dataset_path, phase + '_keypoints', os.path.basename(f))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for img_name in sorted(glob.glob(os.path.join(f, '*.jpg'))):
        img = io.imread(img_name)
        dets = detector(img, 1)
        if len(dets) > 0:
            shape = predictor(img, dets[0])
            points = np.empty([68, 2], dtype=int)
            for b in range(68):
                points[b,0] = shape.part(b).x
                points[b,1] = shape.part(b).y

            save_name = os.path.join(save_path, os.path.basename(img_name)[:-4] + '.txt')
            np.savetxt(save_name, points, fmt='%d', delimiter=',')
