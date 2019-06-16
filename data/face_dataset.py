import os.path
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
import cv2
from skimage import feature

from data.base_dataset import BaseDataset, get_img_params, get_transform, get_video_params, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from data.keypoint2img import interpPoints, drawEdge

class FaceDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot                
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_keypoints')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_img')
        
        self.A_paths = sorted(make_grouped_dataset(self.dir_A))
        self.B_paths = sorted(make_grouped_dataset(self.dir_B))    
        check_path_valid(self.A_paths, self.B_paths)

        self.init_frame_idx(self.A_paths)
        self.scale_ratio = np.array([[0.9, 1], [1, 1], [0.9, 1], [1, 1.1], [0.9, 0.9], [0.9, 0.9]])#np.random.uniform(0.9, 1.1, size=[6, 2])
        self.scale_ratio_sym = np.array([[1, 1], [0.9, 1], [1, 1], [0.9, 1], [1, 1], [1, 1]]) #np.random.uniform(0.9, 1.1, size=[6, 2])
        self.scale_shift = np.zeros((6, 2)) #np.random.uniform(-5, 5, size=[6, 2])

    def __getitem__(self, index):
        A, B, I, seq_idx = self.update_frame_idx(self.A_paths, index)        
        A_paths = self.A_paths[seq_idx]
        B_paths = self.B_paths[seq_idx]
        n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(A_paths), self.frame_idx)
        
        B_img = Image.open(B_paths[start_idx]).convert('RGB')
        B_size = B_img.size
        points = np.loadtxt(A_paths[start_idx], delimiter=',')
        is_first_frame = self.opt.isTrain or not hasattr(self, 'min_x')
        if is_first_frame: # crop only the face region
            self.get_crop_coords(points, B_size)
        params = get_img_params(self.opt, self.crop(B_img).size)        
        transform_scaleA = get_transform(self.opt, params, method=Image.BILINEAR, normalize=False)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        transform_scaleB = get_transform(self.opt, params)
        
        # read in images        
        frame_range = list(range(n_frames_total)) if self.A is None else [self.opt.n_frames_G-1]        
        for i in frame_range:
            A_path = A_paths[start_idx + i * t_step]
            B_path = B_paths[start_idx + i * t_step]                    
            B_img = Image.open(B_path)
            Ai, Li = self.get_face_image(A_path, transform_scaleA, transform_label, B_size, B_img)
            Bi = transform_scaleB(self.crop(B_img))
            A = concat_frame(A, Ai, n_frames_total)
            B = concat_frame(B, Bi, n_frames_total)
            I = concat_frame(I, Li, n_frames_total)
        
        if not self.opt.isTrain:
            self.A, self.B, self.I = A, B, I
            self.frame_idx += 1
        change_seq = False if self.opt.isTrain else self.change_seq
        return_list = {'A': A, 'B': B, 'inst': I, 'A_path': A_path, 'change_seq': change_seq}
                
        return return_list

    def get_image(self, A_path, transform_scaleA):
        A_img = Image.open(A_path)                
        A_scaled = transform_scaleA(self.crop(A_img))
        return A_scaled

    def get_face_image(self, A_path, transform_A, transform_L, size, img):
        # read face keypoints from path and crop face region
        keypoints, part_list, part_labels = self.read_keypoints(A_path, size)

        # draw edges and possibly add distance transform maps
        add_dist_map = not self.opt.no_dist_map
        im_edges, dist_tensor = self.draw_face_edges(keypoints, part_list, transform_A, size, add_dist_map)
        
        # canny edge for background
        if not self.opt.no_canny_edge:
            edges = feature.canny(np.array(img.convert('L')))        
            edges = edges * (part_labels == 0)  # remove edges within face
            im_edges += (edges * 255).astype(np.uint8)
        edge_tensor = transform_A(Image.fromarray(self.crop(im_edges)))

        # final input tensor
        input_tensor = torch.cat([edge_tensor, dist_tensor]) if add_dist_map else edge_tensor
        label_tensor = transform_L(Image.fromarray(self.crop(part_labels.astype(np.uint8)))) * 255.0
        return input_tensor, label_tensor

    def read_keypoints(self, A_path, size):        
        # mapping from keypoints to face part 
        part_list = [[list(range(0, 17)) + list(range(68, 83)) + [0]], # face
                     [range(17, 22)],                                  # right eyebrow
                     [range(22, 27)],                                  # left eyebrow
                     [[28, 31], range(31, 36), [35, 28]],              # nose
                     [[36,37,38,39], [39,40,41,36]],                   # right eye
                     [[42,43,44,45], [45,46,47,42]],                   # left eye
                     [range(48, 55), [54,55,56,57,58,59,48]],          # mouth
                     [range(60, 65), [64,65,66,67,60]]                 # tongue
                    ]
        label_list = [1, 2, 2, 3, 4, 4, 5, 6] # labeling for different facial parts        
        keypoints = np.loadtxt(A_path, delimiter=',')
        
        # add upper half face by symmetry
        pts = keypoints[:17, :].astype(np.int32)
        baseline_y = (pts[0,1] + pts[-1,1]) / 2
        upper_pts = pts[1:-1,:].copy()
        upper_pts[:,1] = baseline_y + (baseline_y-upper_pts[:,1]) * 2 // 3
        keypoints = np.vstack((keypoints, upper_pts[::-1,:]))  

        # label map for facial part
        w, h = size
        part_labels = np.zeros((h, w), np.uint8)
        for p, edge_list in enumerate(part_list):                
            indices = [item for sublist in edge_list for item in sublist]
            pts = keypoints[indices, :].astype(np.int32)
            cv2.fillPoly(part_labels, pts=[pts], color=label_list[p]) 

        # move the keypoints a bit
        if not self.opt.isTrain and self.opt.random_scale_points:
            self.scale_points(keypoints, part_list[1] + part_list[2], 1, sym=True)
            self.scale_points(keypoints, part_list[4] + part_list[5], 3, sym=True)
            for i, part in enumerate(part_list):
                self.scale_points(keypoints, part, label_list[i]-1)

        return keypoints, part_list, part_labels

    def draw_face_edges(self, keypoints, part_list, transform_A, size, add_dist_map):
        w, h = size
        edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
        # edge map for face region from keypoints
        im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
        dist_tensor = 0
        e = 1                
        for edge_list in part_list:
            for edge in edge_list:
                im_edge = np.zeros((h, w), np.uint8) # edge map for the current edge
                for i in range(0, max(1, len(edge)-1), edge_len-1): # divide a long edge into multiple small edges when drawing
                    sub_edge = edge[i:i+edge_len]
                    x = keypoints[sub_edge, 0]
                    y = keypoints[sub_edge, 1]
                                    
                    curve_x, curve_y = interpPoints(x, y) # interp keypoints to get the curve shape                    
                    drawEdge(im_edges, curve_x, curve_y)
                    if add_dist_map:
                        drawEdge(im_edge, curve_x, curve_y)
                                
                if add_dist_map: # add distance transform map on each facial part
                    im_dist = cv2.distanceTransform(255-im_edge, cv2.DIST_L1, 3)    
                    im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
                    im_dist = Image.fromarray(im_dist)
                    tensor_cropped = transform_A(self.crop(im_dist))                    
                    dist_tensor = tensor_cropped if e == 1 else torch.cat([dist_tensor, tensor_cropped])
                    e += 1

        return im_edges, dist_tensor

    def get_crop_coords(self, keypoints, size):                
        min_y, max_y = keypoints[:,1].min(), keypoints[:,1].max()
        min_x, max_x = keypoints[:,0].min(), keypoints[:,0].max()                
        xc = (min_x + max_x) // 2
        yc = (min_y*3 + max_y) // 4
        h = w = (max_x - min_x) * 2.5        
        xc = min(max(0, xc - w//2) + w, size[0]) - w//2
        yc = min(max(0, yc - h//2) + h, size[1]) - h//2
        min_x, max_x = xc - w//2, xc + w//2
        min_y, max_y = yc - h//2, yc + h//2        
        self.min_y, self.max_y, self.min_x, self.max_x = int(min_y), int(max_y), int(min_x), int(max_x)        

    def crop(self, img):
        if isinstance(img, np.ndarray):
            return img[self.min_y:self.max_y, self.min_x:self.max_x]
        else:
            return img.crop((self.min_x, self.min_y, self.max_x, self.max_y))

    def scale_points(self, keypoints, part, index, sym=False):
        if sym:
            pts_idx = sum([list(idx) for idx in part], [])
            pts = keypoints[pts_idx]
            ratio_x = self.scale_ratio_sym[index, 0]
            ratio_y = self.scale_ratio_sym[index, 1]
            mean = np.mean(pts, axis=0)
            mean_x, mean_y = mean[0], mean[1]
            for idx in part:
                pts_i = keypoints[idx]
                mean_i = np.mean(pts_i, axis=0)
                mean_ix, mean_iy = mean_i[0], mean_i[1]
                new_mean_ix = (mean_ix - mean_x) * ratio_x + mean_x
                new_mean_iy = (mean_iy - mean_y) * ratio_y + mean_y
                pts_i[:,0] = (pts_i[:,0] - mean_ix) + new_mean_ix
                pts_i[:,1] = (pts_i[:,1] - mean_iy) + new_mean_iy
                keypoints[idx] = pts_i

        else:            
            pts_idx = sum([list(idx) for idx in part], [])
            pts = keypoints[pts_idx]
            ratio_x = self.scale_ratio[index, 0]
            ratio_y = self.scale_ratio[index, 1]
            mean = np.mean(pts, axis=0)
            mean_x, mean_y = mean[0], mean[1]            
            pts[:,0] = (pts[:,0] - mean_x) * ratio_x + mean_x + self.scale_shift[index, 0]
            pts[:,1] = (pts[:,1] - mean_y) * ratio_y + mean_y + self.scale_shift[index, 1]
            keypoints[pts_idx] = pts

    def __len__(self):
        if self.opt.isTrain:
            return len(self.A_paths)
        else:
            return sum(self.frames_count)

    def name(self):
        return 'FaceDataset'