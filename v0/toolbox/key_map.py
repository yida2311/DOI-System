import numpy as np 
import os
import cv2
from skimage.measure import label
from PIL import Image
import math
import pdb
import glob
import scipy
import json
import pandas as pd
from tqdm import tqdm
import copy


class KeyMap(object):
    #def __init__(self, map_path, save_dir, cfg):
    def __init__(self, map_path):
        """数据结构
            contour, boundary: [[[x1,y1]], [[x2,y2]], ...]
            point: [x, y]
        """
        super(KeyMap, self).__init__()
        #self.cfg = cfg
        #self.save_dir = save_dir
        self.name = map_path.split('/')[-1]
        self.map = cv2.imread(map_path)  # H x W x 3, BGR
        #print(self.map.dtype)
        self.map0 = cv2.cvtColor(self.map, cv2.COLOR_BGR2RGB)
        #print(self.map.dtype)
        self.map = np.around(cv2.blur(self.map0,(61,61))/255)*255
        self.map = self.map.astype(np.uint8)
        #print(self.map.dtype)
        #blurred_path = os.path.join(os.path.join(self.save_dir, 'blurred'), self.name)
        #blurred = cv2.cvtColor(self.map, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(blurred_path, blurred)
        self.map_path = map_path
        #self.map_name = map_path.split('/')[-1].split('_')[1] + '.png'
        self.map_name = map_path.split('/')[-1] + '.png'
        self.size = self.map.shape  # h x w x 3
        self.padding = 3
        self.size = [self.size[0]+2*self.padding, self.size[1]+2*self.padding, self.size[2]]

        self.normal = self.map[:,:,0]
        self.normal = self.pad_mask(self.normal, self.padding)
        self.mucosa = self.map[:,:,1]
        self.mucosa = self.pad_mask(self.mucosa, self.padding)
        self.tumor = self.map[:,:,2]
        self.tumor = self.pad_mask(self.tumor, self.padding)
        self.contour_tumor = self.get_contour(self.tumor)

        #########################################################################################
        ##                                 A. mask generation                                  ##
        #########################################################################################
        self.mask = self.generate_mask()
        #mask_path = os.path.join(os.path.join(self.save_dir, 'mask'), self.name)
        #cv2.imwrite(mask_path, self.mask)

        #########################################################################################
        ##                                 B. Split Contour_top                                ##
        #########################################################################################
        self.generate_split_contours()
        mask_split = np.zeros(self.size, dtype='uint8')
        color_map = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        #split_contours_path = os.path.join(os.path.join(self.save_dir, 'split_contours'), self.name)
        for idx, cnt in enumerate(self.split_cnts):
            cls_name = self.split_class[idx]
            value = color_map[cls_name]
            cnt = np.array(cnt, dtype=np.int32)
            cnt = cnt.reshape((-1,1,2))
            cv2.polylines(mask_split, [cnt], False, value, 3)
        mask_split = cv2.cvtColor(mask_split, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(split_contours_path, mask_split)

        #########################################################################################
        ##                                 C. Analyze Split_Contours                           ##
        #########################################################################################
        self.cells = self.analyze_split_contours(self.split_cnts, self.split_class)
        self.min_distance=0
        for cell in self.cells:
            self.cell=cell
            mask_cell = np.zeros(self.size, dtype='uint8')
            # grayscale = [85, 170, 255]
            #cell_path = os.path.join(os.path.join(self.save_dir, 'cell'), self.name)
            for idx, cnt_idx in enumerate(self.cell):
                cnt = self.split_cnts[cnt_idx]
                if idx == 0 or idx == len(self.cell)-1:
                    value = [0, 255, 0]
                else:
                    value = [0, 0, 255]
                cnt = np.array(cnt, dtype=np.int32)
                cnt = cnt.reshape((-1,1,2))
                cv2.polylines(mask_cell, [cnt], False, value, 3)
            mask_cell = cv2.cvtColor(mask_cell, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(cell_path, mask_cell)

        #########################################################################################
        ##                                 D. Confirm Referenced Mucosa                        ##
        #########################################################################################
            self.start_points, self.loc, self.contour_mucosa_sel = self.analyze_cell(self.cell, self.split_cnts, self.mucosa)
            mask_mucosa = np.zeros(self.size, dtype='uint8')
            #mucosa_path = os.path.join(os.path.join(self.save_dir, 'mucosa'), self.name)
            for idx, cnt in enumerate(self.contour_mucosa_sel):
                cnt = np.array(cnt, dtype=np.int32)
                cnt = cnt.reshape((-1,1,2))
                cv2.polylines(mask_mucosa, [cnt], False, [0, 255, 0], 3)

                spt = self.start_points[idx]
                spt = (spt[0], spt[1])
                cv2.circle(mask_mucosa,spt, 4, [255, 255, 255], -1)
            mask_mucosa = cv2.cvtColor(mask_mucosa, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(mucosa_path, mask_mucosa)

        #########################################################################################
        ##                                 E. Search Keypoints                                 ##
        #########################################################################################
            key_points = self.search_keypoint()
            distance=self._calculate_distance_between_points(key_points[0],key_points[1])
            if distance > self.min_distance:
                self.max_distance = distance
                self.key_points = key_points
            

        #########################################################################################
        ##                                 F. DOI Calculation                                 ##
        #########################################################################################
        self.doi, self.key_point_tumor = self.calculate_doi(self.key_points)
        self.foot=self.getFootPoint([self.key_point_tumor[0],self.key_point_tumor[1]], [self.key_points[0][0],self.key_points[0][1]], [self.key_points[1][0],self.key_points[1][1]])


    ###########################################################################################################
    ###########################################################################################################
    def get_mask_size(self):
        return self.map0.shape[0], self.map0.shape[1]
    def pad_mask(self, img, padding=3):
        """ 对mask进行填充， 以便于求边界和轮廓 """
        size = img.shape
        size_padded = (size[0]+2*padding, size[1]+2*padding)
        img_padded = np.zeros(size_padded, dtype='uint8')
        img_padded[padding:size[0]+padding, padding:size[1]+padding] = img
        return img_padded
    
    def get_contour(self, img):
        """
        img: 单通道图像
        save_dir: optional
        """
        _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

    def neighbor_four(self, point):
        """获取点的四邻域"""
        neighbor = []
        neighbor.append([point[0]-1, point[1]])
        neighbor.append([point[0]+1, point[1]])
        neighbor.append([point[0], point[1]-1])
        neighbor.append([point[0], point[1]+1])
        return neighbor
    
    def _calculate_distance_between_points(self, pt1, pt2):
        """计算两点之间最短距离"""
        return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    
    def _calculate_distance_between_contours_and_point(self, cnts, pt):
        """计算一个点到多个轮廓的最短距离"""
        min_distance = 10**12
        for cnt in cnts:
            distance, _ = self._calculate_distance_between_contour_and_point(cnt, pt)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def _calculate_distance_between_contour_and_point(self, cnt, pt):
        """计算点到一个轮廓的距离"""
        min_distance = 10**12
        pt_near = cnt[0][0]
        for pt_r in cnt:
            distance = self._calculate_distance_between_points(pt, pt_r[0])
            if distance < min_distance:
                min_distance = distance
                pt_near = pt_r[0]
        return min_distance, pt_near

    def _calculate_distance_between_line_and_point(self, line, point):
        """计算点到直线之间的距离"""
        A, B, C = line
        return abs(A*point[0] + B*point[1] + C) / math.sqrt(A**2 + B**2)

    def search_position_in_contour(self, cnt, pt):
        """搜索点在轮廓中的位置"""
        for idx, new_pt in enumerate(cnt):
            if (new_pt[0]==pt).all():
                return idx

    def get_contour_area(self, contour):
        """计算轮廓的面积"""
        return cv2.contourArea(contour)

    def get_contour_perimeter(self, contour):
        """计算轮廓的估计周长"""
        hull = cv2.convexHull(contour)  #　得到轮廓的凸包
        length = cv2.arcLength(hull, True)
        return length

    def calculate_product_vector(self, v1, v2):
        """计算两个向量的内积"""
        return v1[0]*v2[0] + v1[1]*v2[1]
    
    def normalize_vector(self, v):
        """向量标准化"""
        length = math.sqrt(v[0]**2 + v[1]**2)
        return [c/length for c in v]

    def contours_to_mask(self, cnts):
        """将多个轮廓转化成mask"""
        base = np.zeros(self.size[:2], dtype='uint8')
        for cnt in cnts:
            cv2.polylines(base, [cnt], True, 255, 1)
            cv2.fillPoly(base, [cnt], 255)
        return base

    def calculate_line_by_points(self, pt1, pt2):
        """
        Ax + By + C = 0
        """
        A = pt1[1] - pt2[1]
        B = -(pt1[0] - pt2[0])
        C = pt1[0]*pt2[1] - pt2[0]*pt1[1]
        return A, B, C


    #########################################################################################
    ##                                 A. mask generation                                  ##
    #########################################################################################
    def generate_mask(self):
        """ 生成包含各类组织的mask， 同时去除不必要的正常组织，避免干扰 """
        # 获取包含各类组织的mask，其中background=0, normal =55, mucosa=155, tumor=255
        self.mask = self.combine_mask(self.normal, self.mucosa, self.tumor)
        # cv2.imwrite('/home/ldy/mask.png', self.mask)
        # b = copy.deepcopy(maskself
        self.mask = self.fill_bgd_holes(self.mask, value=40)
        self.normal_filt = self.filt_normal(self.normal, area_thresh=40, bd_thresh=100)
        self.mask = self.combine_mask(self.normal_filt, self.mucosa, self.tumor)
        self.mask = self.fill_bgd_holes(self.mask, value=40)
        self.tumor_filt = self.filt_tumor(self.tumor, area_thresh=10000)
        self.mucosa_filt = self.filt_mucosa(self.mucosa, area_thresh=10000)
        self.mask = self.combine_mask(self.normal_filt, self.mucosa_filt, self.tumor_filt)
        return self.mask
    
    def combine_mask(self, normal, mucosa, tumor):
        """ 将normal, mucosa, tumor结合起来生成mask """
        mask = np.zeros(self.size[:2], dtype='uint8')
        mask = mask + np.array(normal//3, dtype='uint8')
        mask = mask + np.array(mucosa//3 * 2, dtype='uint8')
        mask = mask + np.array(tumor, dtype='uint8')
        return mask

    def fill_bgd_holes(self, mask, value):
        """ 在mask中， 灰度值为0的区域分为两类：组织外背景和组织内空洞。为避免干扰，需要将组织内空洞填充为另一灰度值，区别于背景"""
        _, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)  # 将mask分成组织和背景两部分，组织为255，背景为0
        thresh = 255 - thresh
        # 将背景各region打上标签
        labeled_img, num = label(thresh, neighbors=8, background=0, return_num=True)

        hole_mask = np.zeros(self.size[:2], dtype='uint8')
        for i in range(1, num+1):
            hole = (labeled_img == i)
            # 计算标签为i的region在图像边界的像素点的个数，如果大于0，则表示其为组织外背景；否则，为hole
            count = np.sum(hole[0, :]) + np.sum(hole[:, 0]) + np.sum(hole[self.size[0]-1, :]) + np.sum(hole[:,self.size[1]-1])
            if count > 0:
                continue
            else:
                hole_mask += hole
        return mask + value*hole_mask
    
    def getFootPoint(self,point, line_p1, line_p2):
        """
        @point, line_p1, line_p2 : [x, y, z]
        """
        x0 = point[0]
        y0 = point[1]
    

        x1 = line_p1[0]
        y1 = line_p1[1]
    

        x2 = line_p2[0]
        y2 = line_p2[1]
    

        k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) ) / \
        ((x2 - x1) ** 2 + (y2 - y1) ** 2 )*1.0

        xn = k * (x2 - x1) + x1
        yn = k * (y2 - y1) + y1
        

        return (int(xn), int(yn))
        
    def filt_normal(self, normal, area_thresh, bd_thresh):
        """部分正常组织的region对于任务没有帮助，反而形成干扰，需要去除"""
        _, thresh = cv2.threshold(normal, 127, 225, cv2.THRESH_BINARY)
        labeled_img, num = label(thresh, neighbors=4, background=0, return_num=True)

        # 计算最大连通域
        max_label = 1
        max_num = 0
        for i in range(1, num+1): 
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
        mask = (labeled_img == max_label)

        for i in range(1, num+1):
            if i == max_label:
                continue
            region = normal * (labeled_img == i)
            # 确定该region是否满足要求
            if self.check_normal_region_condition(region, area_thresh, bd_thresh):
                mask += (labeled_img == i)
        return normal * mask
    
    def check_normal_region_condition(self, region, area_thresh, bd_thresh):
        """
        排除标准：
            １．只与background, mucosa相邻；只与background, tumor相邻
            ２．同时与background, mucosa, tumor相邻，若面积过小或者相邻边界过短，去除　
        return:
            True: 保留 
            False: 去除
        """
        region_cnt = self.get_contour(region)[0]
        # flag_X 表示是否与X相邻
        flag_bgd = False
        flag_mucosa = False
        flag_tumor = False
        len_bd = 0 # 与mucosa, tumor的边界

        for pt in region_cnt:
            pt = pt[0]
            if flag_bgd & flag_mucosa & flag_tumor:
                break
            for nb in self.neighbor_four(pt):
                value = self.mask[nb[1], nb[0]]
                if value == 0:
                    flag_bgd = True
                elif value == 170:
                    flag_mucosa = True
                    len_bd += 1
                    break
                elif value == 255:
                    flag_tumor = True
                    len_bd += 1 
                    break
        if (flag_bgd & flag_mucosa & (~flag_tumor)) or (flag_bgd & flag_tumor & (~flag_mucosa)):
            return False
        elif flag_bgd & flag_mucosa & flag_tumor:
            area = self.get_contour_area(region_cnt)
            if area > area_thresh or len_bd > bd_thresh:
                return True
            else:
                return False
        else:
            return True
    
    def filt_tumor(self, tumor, area_thresh):
        """去除只与背景和黏膜相邻的肿瘤"""
        _, thresh = cv2.threshold(tumor, 127, 225, cv2.THRESH_BINARY)
        labeled_img, num = label(thresh, neighbors=4, background=0, return_num=True)

        mask = (labeled_img == -1)
        
        for i in range(1, num+1):
            region = tumor * (labeled_img == i)
            if self.check_tumor_region_condition(region, area_thresh):
                mask += (labeled_img == i)
        return tumor * mask

    def check_tumor_region_condition(self, region, area_thresh):
        """
        排除标准：只同时与黏膜和背景相邻或者只与黏膜相邻的排除
        """
        region_cnt = self.get_contour(region)[0]
        # flag_X 表示是否与X相邻
        flag_mucosa = False
        flag_normal = False
        flag_other = False #　40

        for pt in region_cnt:
            pt = pt[0]
            if flag_normal or flag_other:
                break
            for nb in self.neighbor_four(pt):
                value = self.mask[nb[1], nb[0]]
                if value == 40:
                    flag_other = True
                    break
                elif value == 85:
                    flag_normal = True
                    break
                elif value == 170:
                    flag_mucosa = True
        area = self.get_contour_area(region_cnt)
        #print('tumor',area)
        if (flag_mucosa & (~flag_normal) & (~flag_other)) or area < area_thresh:
            return False
        else:
            return True
        
    def filt_mucosa(self, mucosa, area_thresh):
        """去除面积太小的肿瘤块"""
        _, thresh = cv2.threshold(mucosa, 127, 225, cv2.THRESH_BINARY)
        labeled_img, num = label(thresh, neighbors=4, background=0, return_num=True)

        mask = (labeled_img == -1)
        passed = 0
        areas = []
        for i in range(1, num+1):
            region = mucosa * (labeled_img == i)
            region_cnt = self.get_contour(region)[0]
            areas.append(self.get_contour_area(region_cnt))
        areas.sort(reverse=True)
        #print(areas)
        area_thresh = min(areas[1]*0.8,area_thresh)
        for i in range(1, num+1):
            region = mucosa * (labeled_img == i)
            #print(i,passed)
            if self.check_mucosa_region_condition(region, area_thresh) or (i==num and passed==1) or (i==num-1 and passed==0):
                mask += (labeled_img == i)
                passed += 1
        return mucosa * mask

    def check_mucosa_region_condition(self, region, area_thresh):
        """
        排除标准：面积
        """
        region_cnt = self.get_contour(region)[0]
        area = self.get_contour_area(region_cnt)
        #print('mucosa',area)
        if area <= area_thresh:
            return False
        else:
            return True
                            

    #########################################################################################
    ##                                 B. Split Contour_top                                ##
    #########################################################################################
    def generate_split_contours(self):
        contour_top = self.get_mask_top_contour(self.mask)
        self.split_cnts, self.split_class = self.split_top_contour(contour_top, self.mask)

    def get_mask_top_contour(self, mask):
        """选出最大的组织并返回其轮廓"""
        _, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        labeled_img, num = label(thresh, neighbors=8, background=0, return_num=True)

        # 计算最大连通域
        mask_label = 1
        max_num = 0
        for i in range(1, num+1):
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
        mask_top = (labeled_img == max_label) * mask
        return self.get_contour(mask_top)[0]

    def split_top_contour(self, top_cnt, mask):
        """将轮廓按照所在位置的类别分解"""
        split_cnts = []  # 分解的轮廓
        split_class = []  # 分解轮廓对应的类别
        boundary = []
        state = mask[top_cnt[0][0][1], top_cnt[0][0][0]]
        for pt in top_cnt:
            if mask[pt[0][1], pt[0][0]] == state:
                boundary.append(pt)
            else:
                split_cnts.append(boundary)
                if state == 85:  
                    split_class.append(0)
                elif state == 170:
                    split_class.append(1)
                elif state == 255:
                    split_class.append(2)
                else:
                    raise ValueError('wrong value {} in point {}'.format(state, pt[0]))
                
                state = mask[pt[0][1], pt[0][0]]
                boundary = [pt]
        if len(boundary) > 0:
            split_cnts.append(boundary)
            if state == 85:  
                split_class.append(0)
            elif state == 170:
                split_class.append(1)
            elif state == 255:
                split_class.append(2)
            else:
                raise ValueError('wrong value {} in point {}'.format(state, pt[0]))
        
        if split_class[0] == split_class[-1]:
            resid_cnt = split_cnts[0]
            del split_cnts[0]
            del split_class[0]
            split_cnts[-1] = split_cnts[-1] + resid_cnt 
        
        split_cnts, split_class = self._delete_tumor_boundary_in_edge(split_cnts, split_class)
        return split_cnts, split_class
    
    def check_boundary_in_edge(self, bd):
        """检测边界是否在图像边缘，若超过３/4则返回True"""
        len_bd = len(bd)
        bd_np = np.array(bd).reshape((-1, 2))
        edge_points = 0
        for i in range(3, 10):
            edge_points += np.sum(bd_np[:, 0] == i) + np.sum(bd_np[:, 1] == i) + np.sum(bd_np[:, 0] == (self.size[1]-i)) + np.sum(bd_np[:, 1] == (self.size[0]-i))
        if edge_points > 3 * len_bd/4:
            return True
        else:
            return False
    
    def _delete_tumor_boundary_in_edge(self, cnts, classes):
        """去除在图像边缘的肿瘤的边界"""
        num = len(classes)
        for i in range(num):
            if classes[i] == 2:
                if self.check_boundary_in_edge(cnts[i]):
                    if classes[i-1] == classes[i+1]:
                        cnts[i-1] = cnts[i-1] + cnts[i+1]
                        del cnts[i:i+2]
                        del classes[i:i+2]
                    else:
                        del cnts[i]
                        del classes[i]
                    return self._delete_tumor_boundary_in_edge(cnts, classes)
        return cnts, classes


    #########################################################################################
    ##                                 C. Analyze Split_Contours                           ##
    #########################################################################################
    def analyze_split_contours(self, split_cnts, split_class):
        cells = self.translate_split_contours_to_cells(split_cnts, split_class)
        redund=1600
        steps = []
        min_step_cells=[]
        for cell in cells:
            if self.delete_cell(cell, split_cnts, step_thresh=10):
                step = 10 ** 12
            else:
                step = self.measure_cell_step(cell, split_cnts, split_class)
            steps.append(step)
        for i in range(len(steps)): 
            if steps[i]<min(steps)+redund:
                min_step_cells.append(cells[i])
        return min_step_cells

    def translate_split_contours_to_cells(self, split_cnts, split_class):
        """从split_contours中选出符合黏膜表面分布的cells:[mucosa, tumor, (tumor), mucosa]"""
        num_mucosa = len([c for c in split_class if c == 1])
        num_tumor = len([c for c in split_class if c == 2])
        
        assert num_mucosa > 1 and num_tumor > 0

        cells = [] # cell: [mucosa, tumor, (tumor), mucosa]
        cell = []

        flag_mucosa = False
        flag_tumor = False
        tumor_latent = -1
        start_cnts = []
        
        for idx, cls_name in enumerate(split_class):
            cnt = [split_cnts[idx]]
            if cls_name == 1:
                if flag_mucosa:
                    if flag_tumor: # [mucosa, tumor, mucosa]
                        flag_tumor = False
                        if tumor_latent == -1: # [mucosa, tumor, mucosa]
                            cell.append(idx)
                            cells.append(cell)
                            cell = [idx]
                        else:  # [mucosa, tumor, tumor_latent, mucosa]
                            cell.append(tumor_latent)
                            cell.append(idx)
                            cells.append(cell)
                            cell = [idx]
                            tumor_latent = -1
                    else: # [mucosa, mucosa]
                        cell = [idx]
                else: # [mucosa]  START STATE
                    flag_mucosa = True
                    flag_tumor = False
                    cell.append(idx)
                    start_cnts.append(idx)
            elif cls_name == 2:
                if flag_mucosa:
                    if flag_tumor: # [mucosa, tumor, tumor_latent]
                        tumor_latent = idx
                    else: # [mucosa, tumor]
                        flag_tumor = True
                        cell.append(idx)
                else:  # START STATE
                    if flag_tumor: # [tumor, tumor_latent]
                        start_cnts.append(idx)
                    else:  # [tumor]
                        flag_tumor = True
                        start_cnts.append(idx)
        if tumor_latent != -1:
            cell.append(tumor_latent)
        new_cell = cell + start_cnts
        if len(new_cell) > 2:
            if len(new_cell) > 4:
                new_cell = new_cell[:2] + new_cell[-2:]
            cells.append(new_cell)
        assert len(cells) > 0
        return cells
    
    def delete_cell(self, cell, split_cnts, step_thresh):
        """
        删除不满足要求的cell：即肿瘤边界过短的cell
        return:
            True: 删除
            False: 保留
        """
        if len(cell) == 3:
            step = len(split_cnts[cell[1]])
        elif len(cell) == 4:
            step = len(split_cnts[cell[1]]) + len(split_cnts[cell[2]])
        else:
            raise ValueError(len(cell))
        
        if step < step_thresh:
            return True
        else:
            return False

    def measure_cell_step(self, cell, split_contours, split_class):
        """计算cell中两边黏膜到对应肿瘤的步长之和"""
        num_cnt = len(split_contours)
        mucosa_idx_1 = cell[0]
        mucosa_idx_2 = cell[-1]
        if len(cell) == 3:
            tumor_idx = cell[1]
            step1 = self._calculate_cnts_step(mucosa_idx_1, tumor_idx, split_contours)
            #step1 = self._find_mucosa_step(tumor_idx, split_contours, split_class)
            step2 = self._calculate_cnts_step(tumor_idx, mucosa_idx_2, split_contours)
            #step2 = step1
            #step3 = 10000-self._calculate_cnts_step(mucosa_idx_1, mucosa_idx_2, split_contours)
        elif len(cell) == 4:
            tumor_idx_1 = cell[1]
            tumor_idx_2 = cell[2]
            step1 = self._calculate_cnts_step(mucosa_idx_1, tumor_idx_1, split_contours)
            #step1 = self._find_mucosa_step(tumor_idx_1, split_contours, split_class)
            step2 = self._calculate_cnts_step(tumor_idx_2, mucosa_idx_2, split_contours)
            #step2 = self._find_mucosa_step(tumor_idx_2, split_contours, split_class)
            #step3 = 10000-self._calculate_cnts_step(mucosa_idx_1, mucosa_idx_2, split_contours) 
        else:
            raise ValueError
        step = step1 + step2 
        #print('cell',step)
        return step

    def _calculate_cnts_step(self, idx1, idx2, split_contours):
        """计算对应黏膜和肿瘤索引之间的步长"""
        num_cnt = len(split_contours)
        #print(idx1,idx2,num_cnt)
        if abs((idx2 - idx1) % num_cnt) == 1:
            step = 0
        elif (idx2 - idx1) % num_cnt == 2:
            step = len(split_contours[(idx1+1)%num_cnt])
        elif (idx1 - idx2) % num_cnt == 2:
            step = len(split_contours[(idx2+1)%num_cnt])
        else:
            raise ValueError('wrong:{}/{}'.format((idx2 - idx1) % num_cnt, num_cnt))
        #print('cnts',step)
        return step
    
    def _find_mucosa_step(self, idx, split_contours, split_class):
        """计算对应黏膜到最近肿瘤的步长"""
        print(split_class[idx-2:idx+3])
        num_cnt = len(split_contours)
        step=10**6
        if split_class[(idx+1)%num_cnt]==1 or split_class[(idx-1)%num_cnt]==1:
            step=0
        elif split_class[(idx+2)%num_cnt]==1:
            step=len(split_contours[(idx+1)%num_cnt])
        elif split_class[(idx-2)%num_cnt]==1:
            step=min(len(split_contours[(idx-1)%num_cnt]),step)
        return step

    #########################################################################################
    ##                                 D. Confirm Referenced Mucosa                        ##
    #########################################################################################
    def analyze_cell(self, cell, split_contours, mucosa):
        "“”根据cell获取起始点，所需的黏膜轮廓，及对应方向"""
        cnts = self.get_contour(mucosa)
        mucosa_bd1 = split_contours[cell[0]]
        mucosa_bd2 = split_contours[cell[-1]]

        start_points = [mucosa_bd1[-1][0], mucosa_bd2[0][0]]
        loc = [True, False] # True表示在右边
        contour_mucosa_sel = []

        for spt in start_points:
            min_distance = 10**12
            target_cnt = []
            for cnt in cnts:
                distance = self._calculate_distance_between_contour_and_point(cnt, spt)[0]
                if distance < min_distance:
                    min_distance = distance
                    target_cnt = cnt
            contour_mucosa_sel.append(target_cnt)
        # assert not (contour_mucosa_sel[0] == contour_mucosa_sel[1])
        return start_points, loc, contour_mucosa_sel
    
    #########################################################################################
    ##                                    E. Search Keypoint                               ##
    #########################################################################################
    def search_keypoint(self, alpha=2):
        key_points = []
        for idx, spt in enumerate(self.start_points):
            loc = self.loc[idx]
            contour_mucosa = self.contour_mucosa_sel[idx]
            length = len(contour_mucosa) # 黏膜轮廓的长度
            width = self._calculate_width_of_mucosa(contour_mucosa)
            search_step = int(alpha * width) # 搜索的最大步长
            spt_idx = self.search_position_in_contour(contour_mucosa, spt) #起始点在黏膜的索引
            loc_value = 1 if loc else -1

            for i in range(search_step):
                new_idx = (spt_idx + loc_value * i) % length
                new_point = contour_mucosa[new_idx][0]
                flag_normal = False
                for pt in self.neighbor_four(new_point):
                    value = self.mask[pt[1], pt[0]]
                    if value == 85: # 轮廓邻域存在正常组织
                        for nb in self.neighbor_four(pt):
                            nb_value = self.mask[nb[1], nb[0]]
                            if nb_value == 85:
                                flag_normal = True
                                break
                        if flag_normal:
                            break
                if flag_normal:
                    key_point = new_point
                    break
            if not flag_normal:
                # 在最大搜索步长内都没有找到关键点时
                key_point = new_point
            key_points.append(key_point)
        return key_points
    
    #########################################################################################
    ##                                    F. DOI Calculation                               ##
    #########################################################################################
    def calculate_doi(self, points):
        if self.loc[0]:
            pt_r = points[0]
            pt_l = points[1]
        else:
            pt_r = points[1]
            pt_l = points[0]
        vec_down = [pt_r[1]-pt_l[1], pt_l[0]-pt_r[0]]
        max_distance = -1000
        kpt_tumor = []
        base_line = self.calculate_line_by_points(pt_r, pt_l)
        for cnt in self.contour_tumor:
            for pt in cnt:
                pt = pt[0]
                distance = self._calculate_distance_between_line_and_point(base_line, pt)

                vect_pt = [pt_r[0]+pt_l[0]-2*pt[0], pt_r[1]+pt_l[1]-2*pt[1]]
                if self.calculate_product_vector(vec_down, vect_pt) < 0:
                    distance = - distance
                if distance > max_distance:
                    max_distance = distance
                    kpt_tumor = pt
        return max_distance, kpt_tumor


    #########################################################################################
    ###                                     GT Keypoint Analyze                           ###
    #########################################################################################
    def analyze_kpt_gt(self, kpt_gts):
        kpt_gts = self.identify_kpt_loc(kpt_gts)

        results = []
        for idx, kpt_gt in enumerate(kpt_gts):
            contour_mucosa = self.contour_mucosa_sel[idx]
            spt = self.start_points[idx]
            kpt_new = self._modify_kpt_gt(kpt_gt, contour_mucosa)
            # print(spt, kpt_new, kpt_gt)
            distance = self._calculate_distance_between_points(kpt_new, spt)
            step = self._calculate_step_between_kpt_and_spt(contour_mucosa, kpt_new, spt)
            width = self._calculate_width_of_mucosa(contour_mucosa)

            results.append(dict(distance=distance, step=step, width=width))
        return results
    
    def identify_kpt_loc(self, kpts):
        """调整关键点的左右位置，使得与self.start_point一致"""
        kpt1 = kpts[0]
        kpt2 = kpts[1]
        spt1 = self.start_points[0]
        spt2 = self.start_points[1]
        vec_kpt = [kpt1[0]-kpt2[0], kpt1[1]-kpt2[1]]
        vec_spt = [spt1[0]-spt2[0], spt1[1]-spt2[1]]

        if self.calculate_product_vector(vec_kpt, vec_spt) > 0:
            return kpts
        else:
            return [kpt2, kpt1]


    def _modify_kpt_gt(self, kpt_gt, contour_mucosa):
        """从黏膜轮廓中确定离kpt_gt最近的点"""
        min_distance, kpt_new = self._calculate_distance_between_contour_and_point(contour_mucosa, kpt_gt)
        return kpt_new
    
    def _calculate_step_between_kpt_and_spt(self, contour_mucosa, kpt, spt):
        """计算关键点与起始点之间的距离和步长"""
        kpt_idx = self.search_position_in_contour(contour_mucosa, kpt)
        spt_idx = self.search_position_in_contour(contour_mucosa, spt)
        step = abs(kpt_idx - spt_idx)
        step = min(step, len(contour_mucosa)-step)
        return step
    
    def _calculate_depth_of_kpt(self, mucosa_boundary_with_bgd, kpt):
        """计算关键点的深度"""
        return self._calculate_distance_between_contours_and_point(mucosa_boundary_with_bgd, kpt)
    
    def _calculate_width_of_mucosa(self, contour_mucosa, alpha=2):
        area = self.get_contour_area(contour_mucosa)
        perimeter = self.get_contour_perimeter(contour_mucosa)
        width = area / perimeter * alpha
        # print(area, perimeter)
        return width
        


if __name__ == '__main__':
    slide_info_path = '/Users/Ethan/Downloads/slide_info.json'
    img_dir = 'D:/Academic/Django/projects/DOI/IMAGES/'
    #save_dir = '/Users/Ethan/Downloads/new/globalprediction/fpn_deeposcc_global_test/result/'
    img_list = sorted(glob.glob(img_dir+'*.png'))
    print(img_list)
    with open(slide_info_path, 'r') as f:
        slide_info = json.load(f)
    results = []
    doi_list = []
    const = 0.261324 * 16 / 1000
    # img_list = ['/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC_coarse_padding/mask/_20190404173751_mask.png']
    for idx, img_path in enumerate(img_list):
        print(idx)
        #exmap = KeyMap(img_path, save_dir, config)
        exmap = KeyMap(img_path, save_dir, config)
        kpt_pts = exmap.search_keypoint(alpha=2)
        kpt_pts = exmap.key_points

        slide = '_' + img_path.split('/')[-1].split('_')[1]
        for c in slide_info:
            if str(c['slide']) == str(slide[:-4]):
                kpt_gts = c['keypoints'][:2]
                # kpt_tumor_gt = c['keypoints'][2]
                kpt_gts = [[round(c/16) for c in kpt] for kpt in kpt_gts]
                # kpt_tumor_gt = [round(c/16) for c in kpt_tumor_gt]
                print(kpt_gts)
                break
        else:
            kpt_gts=[[0,0],[0,0]]

        # ## 统计ＤＯＩ
        doi_pt = exmap.doi 
        kpt_gts = exmap.identify_kpt_loc(kpt_gts) 
        # # baseline_gt = exmap.calculate_line_by_points(kpt_gts[0], kpt_gts[1])
        doi_gt, _ = exmap.calculate_doi(kpt_gts) 
        distance_r = exmap._calculate_distance_between_points(kpt_pts[0], kpt_gts[0])
        distance_l = exmap._calculate_distance_between_points(kpt_pts[1], kpt_gts[1])
        doi_result = dict(slide=slide, doi_pt=doi_pt*const, doi_gt=doi_gt*const, doi_dev=abs(doi_gt-doi_pt)*const, distance_r=distance_r*const, distance_l=distance_l*const)
        doi_list.append(doi_result)

        
        ## 关键点可视化
        mask_keypoint = exmap.map0
        for idx, kpt_pt in enumerate(kpt_pts):
            kpt_gt = kpt_gts[idx]
            kpt_gt = (kpt_gt[0], kpt_gt[1])
            kpt_pt = (kpt_pt[0], kpt_pt[1])

            cv2.circle(mask_keypoint, kpt_gt, 25, [255, 0, 255], -1)
            cv2.circle(mask_keypoint, kpt_pt, 25, [255, 255, 255], -1)
        mask_path = os.path.join(os.path.join(save_dir, 'predict2'), slide+'.png')
        print(mask_path)
        mask_keypoint = cv2.cvtColor(mask_keypoint, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_path, mask_keypoint)

        ## 统计关键点（ＧＴ）的特征
        # result = exmap.analyze_kpt_gt(kpt_gts)
        # result[0]['slide'] = slide
        # result[1]['slide'] = slide
        # results += result
      


    ## 统计关键点（ＧＴ）的特征
    # gf = pd.DataFrame(results, columns=['slide', 'distance', 'step', 'width'])
    # gf.to_csv(os.path.join(save_dir, 'keypoint_gt_analyze.csv'), index=False)

    ## 统计ＤＯＩ
    df = pd.DataFrame(doi_list, columns=['slide', 'doi_pt', 'doi_gt','doi_dev', 'distance_r', 'distance_l'])
    df.to_csv(os.path.join(save_dir, 'doi_analyze.csv'), index=False)
