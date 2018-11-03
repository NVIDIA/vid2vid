import os.path
from PIL import Image
import numpy as np
import json
import glob
from scipy.optimize import curve_fit
import warnings

def func(x, a, b, c):    
    return a * x**2 + b * x + c

def linear(x, a, b):
    return a * x + b

def setColor(im, yy, xx, color):
    if len(im.shape) == 3:
        if (im[yy, xx] == 0).all():            
            im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]            
        else:            
            im[yy, xx, 0] = ((im[yy, xx, 0].astype(float) + color[0]) / 2).astype(np.uint8)
            im[yy, xx, 1] = ((im[yy, xx, 1].astype(float) + color[1]) / 2).astype(np.uint8)
            im[yy, xx, 2] = ((im[yy, xx, 2].astype(float) + color[2]) / 2).astype(np.uint8)
    else:
        im[yy, xx] = color[0]

def drawEdge(im, x, y, bw=1, color=(255,255,255), draw_end_points=False):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h-1, y+i))
                xx = np.maximum(0, np.minimum(w-1, x+j))
                setColor(im, yy, xx, color)

        # edge endpoints
        if draw_end_points:
            for i in range(-bw*2, bw*2):
                for j in range(-bw*2, bw*2):
                    if (i**2) + (j**2) < (4 * bw**2):
                        yy = np.maximum(0, np.minimum(h-1, np.array([y[0], y[-1]])+i))
                        xx = np.maximum(0, np.minimum(w-1, np.array([x[0], x[-1]])+j))
                        setColor(im, yy, xx, color)

def interpPoints(x, y):    
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interpPoints(y, x)
        if curve_y is None:
            return None, None
    else:        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")    
            if len(x) < 3:
                popt, _ = curve_fit(linear, x, y)
            else:
                popt, _ = curve_fit(func, x, y)                
                if abs(popt[0]) > 1:
                    return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], (x[-1]-x[0]))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)

def read_keypoints(json_input, size, random_drop_prob=0, remove_face_labels=False, basic_point_only=False):
    with open(json_input, encoding='utf-8') as f:
        keypoint_dicts = json.loads(f.read())["people"]

    edge_lists = define_edge_lists(basic_point_only)
    w, h = size    
    pose_img = np.zeros((h, w, 3), np.uint8)
    for keypoint_dict in keypoint_dicts:    
        pose_pts = np.array(keypoint_dict["pose_keypoints_2d"]).reshape(25, 3)
        face_pts = np.array(keypoint_dict["face_keypoints_2d"]).reshape(70, 3)
        hand_pts_l = np.array(keypoint_dict["hand_left_keypoints_2d"]).reshape(21, 3)
        hand_pts_r = np.array(keypoint_dict["hand_right_keypoints_2d"]).reshape(21, 3)            
        pts = [extract_valid_keypoints(pts, edge_lists) for pts in [pose_pts, face_pts, hand_pts_l, hand_pts_r]]           
        pose_img += connect_keypoints(pts, edge_lists, size, random_drop_prob, remove_face_labels, basic_point_only)
    return pose_img

def extract_valid_keypoints(pts, edge_lists):
    pose_edge_list, _, hand_edge_list, _, face_list = edge_lists
    p = pts.shape[0]
    thre = 0.1 if p == 70 else 0.01
    output = np.zeros((p, 2))    

    if p == 70:   # face
        for edge_list in face_list:
            for edge in edge_list:
                if (pts[edge, 2] > thre).all():
                    output[edge, :] = pts[edge, :2]        
    elif p == 21: # hand        
        for edge in hand_edge_list:            
            if (pts[edge, 2] > thre).all():
                output[edge, :] = pts[edge, :2]
    else:         # pose
        valid = (pts[:, 2] > thre)        
        output[valid, :] = pts[valid, :2]
        
    return output

def connect_keypoints(pts, edge_lists, size, random_drop_prob, remove_face_labels, basic_point_only):
    pose_pts, face_pts, hand_pts_l, hand_pts_r = pts
    w, h = size
    output_edges = np.zeros((h, w, 3), np.uint8)
    pose_edge_list, pose_color_list, hand_edge_list, hand_color_list, face_list = edge_lists
    
    if random_drop_prob > 0 and remove_face_labels:
        # add random noise to keypoints
        pose_pts[[0,15,16,17,18], :] += 5 * np.random.randn(5,2)
        face_pts[:,0] += 2 * np.random.randn()
        face_pts[:,1] += 2 * np.random.randn()

    ### pose    
    for i, edge in enumerate(pose_edge_list):
        x, y = pose_pts[edge, 0], pose_pts[edge, 1]
        if (np.random.rand() > random_drop_prob) and (0 not in x):
            curve_x, curve_y = interpPoints(x, y)                                        
            drawEdge(output_edges, curve_x, curve_y, bw=3, color=pose_color_list[i], draw_end_points=True)

    if not basic_point_only:
        ### hand       
        for hand_pts in [hand_pts_l, hand_pts_r]:     # for left and right hand
            if np.random.rand() > random_drop_prob:
                for i, edge in enumerate(hand_edge_list): # for each finger
                    for j in range(0, len(edge)-1):       # for each part of the finger
                        sub_edge = edge[j:j+2] 
                        x, y = hand_pts[sub_edge, 0], hand_pts[sub_edge, 1]                    
                        if 0 not in x:
                            line_x, line_y = interpPoints(x, y)                                        
                            drawEdge(output_edges, line_x, line_y, bw=1, color=hand_color_list[i], draw_end_points=True)

        ### face
        edge_len = 2
        if (np.random.rand() > random_drop_prob):
            for edge_list in face_list:
                for edge in edge_list:
                    for i in range(0, max(1, len(edge)-1), edge_len-1):             
                        sub_edge = edge[i:i+edge_len]
                        x, y = face_pts[sub_edge, 0], face_pts[sub_edge, 1]
                        if 0 not in x:
                            curve_x, curve_y = interpPoints(x, y)
                            drawEdge(output_edges, curve_x, curve_y, draw_end_points=True)

    return output_edges

def define_edge_lists(basic_point_only):
    ### pose        
    pose_edge_list = []
    pose_color_list = []
    if not basic_point_only:
        pose_edge_list += [[17, 15], [15,  0], [ 0, 16], [16, 18]]       # head
        pose_color_list += [[153,  0,153], [153,  0,102], [102,  0,153], [ 51,  0,153]]

    pose_edge_list += [        
        [ 0,  1], [ 1,  8],                                         # body
        [ 1,  2], [ 2,  3], [ 3,  4],                               # right arm
        [ 1,  5], [ 5,  6], [ 6,  7],                               # left arm
        [ 8,  9], [ 9, 10], [10, 11], [11, 24], [11, 22], [22, 23], # right leg
        [ 8, 12], [12, 13], [13, 14], [14, 21], [14, 19], [19, 20]  # left leg
    ]
    pose_color_list += [
        [153,  0, 51], [153,  0,  0],
        [153, 51,  0], [153,102,  0], [153,153,  0],
        [102,153,  0], [ 51,153,  0], [  0,153,  0],
        [  0,153, 51], [  0,153,102], [  0,153,153], [  0,153,153], [  0,153,153], [  0,153,153],
        [  0,102,153], [  0, 51,153], [  0,  0,153], [  0,  0,153], [  0,  0,153], [  0,  0,153]
    ]

    ### hand
    hand_edge_list = [
        [0,  1,  2,  3,  4],
        [0,  5,  6,  7,  8],
        [0,  9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]
    hand_color_list = [
        [204,0,0], [163,204,0], [0,204,82], [0,82,204], [163,0,204]
    ]

    ### face        
    face_list = [
                 #[range(0, 17)], # face
                 [range(17, 22)], # left eyebrow
                 [range(22, 27)], # right eyebrow
                 [range(27, 31), range(31, 36)], # nose
                 [[36,37,38,39], [39,40,41,36]], # left eye
                 [[42,43,44,45], [45,46,47,42]], # right eye
                 [range(48, 55), [54,55,56,57,58,59,48]], # mouth
                ]
    return pose_edge_list, pose_color_list, hand_edge_list, hand_color_list, face_list