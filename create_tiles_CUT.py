import sys
import os
import numpy as np
from PIL import Image
from PIL import ImageOps
import math
import argparse
import cv2
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import re
import fastai
from fastai.vision.all import *
import datetime
import argparse
from skimage import draw, measure, morphology, filters
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, shape
from shapely.ops import cascaded_union, unary_union
import shapely
import staintools

class extractPatch:

    def __init__(self):
        self.image_location = args.image_location
        self.save_location = args.save_location 
        self.save_image_size = int(args.save_image_size)   # specify image size to be saved (note this is the same for all magnifications)
        self.resamp = args.resamp #1 for 40x, 2 for 20x, 4 for 10x
        self.stain_norm = False #bool(args.stain_norm)
        self.stain_ref_img = args.stain_ref_img




    def parseFolder(self):
        # for subdir, dirs, files in os.walk(self.image_location):
        #     dirs[:] = [d for d in dirs if not d[0] == '.']
        #     files[:] = [f for f in files if f.endswith('.tif')]
            
                    
        #     for file in sorted(files):
        #         tma_id = str.replace(file, '.tif', '')
        #         tma = Image.open(os.path.join(subdir, file))

        #         stain = tma_id.split('_')[1]
        #         stain_tile_path = os.path.join(self.save_location, stain+'_tiles')

        #         #make a low res image to filter for tissue regions
        #         tma_lowres = tma.resize(size=(500,500), resample=Image.LANCZOS)
        #         tissue, he_mask = self.do_mask(tma_lowres, int(tma.size[0]/500))
        #         mask = Image.fromarray(he_mask.astype('bool'))
        #         mask = mask.resize(size=(tma.size[0], tma.size[1]), resample=Image.LANCZOS)
                
                
        #         if not os.path.exists(stain_tile_path):
        #             os.makedirs(stain_tile_path)

        #         print(file)
        #         self.proc_tmaimage(patch=tma, mask=mask, tma_id=tma_id, save_loc=stain_tile_path)
                
                        
        dirlist = [f for f in sorted(os.listdir(self.image_location)) if f.endswith('.tif')]

        if self.stain_norm:
            #Set stuff for stain normalization with staintools
            ref_img = staintools.read_image(self.stain_ref_img)
            METHOD = 'macenko'
            STANDARDIZE_BRIGHTNESS = True
            ref_img = staintools.LuminosityStandardizer.standardize(ref_img)
            normalizer = staintools.StainNormalizer(method=METHOD)
            normalizer.fit(ref_img)

            
        for file in dirlist:
            if self.stain_norm == True:
                img = staintools.read_image(os.path.join(self.image_location, file))
                img = staintools.LuminosityStandardizer.standardize(img)
                img_normalized = normalizer.transform(img)
                tma = Image.fromarray(img_normalized)

            else:
                tma = Image.open(os.path.join(self.image_location, file))
                
            tma_id = str.replace(file,'.tif','')
            #make a low res image to filter for tissue regions
            tma_lowres = tma.resize(size=(500,500), resample=Image.LANCZOS)
            tissue, he_mask = self.do_mask(tma_lowres, int(tma.size[0]/500))
            mask = Image.fromarray(he_mask.astype('bool'))
            mask = mask.resize(size=(tma.size[0], tma.size[1]), resample=Image.LANCZOS)

            
            print(file)
            self.proc_tmaimage(patch=tma, mask=mask, tma_id=tma_id, save_loc=self.save_location)

                                 
    def proc_tmaimage(self, patch, mask, tma_id, save_loc):

        szm = self.resamp * self.save_image_size
        boxSz = self.save_image_size
        stride = szm

        num_subs = math.floor((patch.size[0] - szm) / stride) + 1
        for x in range(1, num_subs + 1):
            for y in range(1, num_subs + 1):

                subpatch = patch.crop(box=(stride * (x - 1) , stride * (y - 1), stride * (x - 1) + szm, stride * (y - 1) + szm))
                maskpatch = mask.crop(box=(stride * (x - 1), stride * (y - 1), stride * (x - 1) + szm, stride * (y - 1) + szm))
                save_coords = str(stride * (x - 1)) + '-' + str(stride * (y - 1))
                patch_20 = subpatch.resize(size=(boxSz, boxSz), resample=Image.LANCZOS)
                mask_20 = maskpatch.resize(size=(boxSz, boxSz), resample=Image.LANCZOS)
                mask_np = np.array(mask_20)
                mask_np = mask_np.astype('uint8')
                mask_save = Image.fromarray(mask_np)
                tumorperc = (np.array(maskpatch) == True).sum()/(szm*szm)
                ws = self.whitespace_check(im=patch_20)
                
                if tumorperc > 0.90 and ws < 0.5: #want 90% or greater of tile to be tissue and less than 50% whitespace
                    tile_savename = tma_id + "_" + save_coords + "_" + "ws-" + '%.2f' % (ws) + '_tumor-'+ '%.2f' % (tumorperc)
                    patch_20.save(os.path.join(save_loc, tile_savename + ".png"))


        return

    
    def do_mask(self,img,lvl_resize):
        ''' create tissue mask '''
        # get he image and find tissue mask
        he = np.array(img)
        he = he[:, :, 0:3]
        heHSV = cv2.cvtColor(he, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(heHSV, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        imagem = cv2.bitwise_not(thresh1)
        tissue_mask = morphology.binary_dilation(imagem, morphology.disk(radius=5))
        tissue_mask = morphology.remove_small_objects(tissue_mask, 1000)
        tissue_mask = ndimage.binary_fill_holes(tissue_mask)


        # create polygons for faster tiling in cancer detection step
        polygons = []
        contours, hier = cv2.findContours(tissue_mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cvals = contour.transpose(0, 2, 1)
            cvals = np.reshape(cvals, (cvals.shape[0], 2))
            cvals = cvals.astype('float64')
            for i in range(len(cvals)):
                cvals[i][0] = np.round(cvals[i][0]*lvl_resize,2)
                cvals[i][1] = np.round(cvals[i][1]*lvl_resize,2)
            try:
                poly = Polygon(cvals)
                if poly.length > 0:
                    polygons.append(Polygon(poly.exterior))
            except:
                pass
        tissue = unary_union(polygons)
        while not tissue.is_valid:
            print('pred_union is invalid, buffering...')
            tissue = tissue.buffer(0)

        return tissue, tissue_mask

    def whitespace_check(self,im):
        bw = im.convert('L')
        bw = np.array(bw)
        bw = bw.astype('float')
        bw=bw/255
        prop_ws = (bw > 0.8).sum()/(bw>0).sum()
        return prop_ws


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_location')
    parser.add_argument('--save_location')
    parser.add_argument('--save_image_size')
    parser.add_argument('--resamp', type=int)
    parser.add_argument('--stain_norm', nargs='?', const=False, type=bool, default=False)
    parser.add_argument('--stain_ref_img', nargs='?', const=None, default=None)
    args = parser.parse_args()
    c = extractPatch()
    c.parseFolder()