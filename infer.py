# import torch libraries
import os
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
import os
import pylab
import numpy as np
import pandas as pd
from PIL import Image
import skimage.io as io
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
# import the utility functions
from model import HED
from dataproc import TestDataset, TestDatasetNonFilter
import glob
import sys
import getopt
import cv2
import csv
from functools import reduce
import pdb
from util import make_txt
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def grayTrans(img):
    img = img.numpy()[0][0] * 255.0
    img = (img).astype(np.uint8)
    return img


def union_intersect(true, pred,threshold=100):
    # Predict matrix, GT matrix vectorize for Intersection 1d , Union 1d, setDiff 1d Calculation
    h,w = true.shape
    nflat=true.ravel().shape

    pred = pred.copy()
    true = true.copy()

    pred=pred.astype(int)
    true=true.astype(int)

    pred[pred<threshold]=0
    pred[pred>=threshold]=255
    true_ravel = true.ravel()
    pred_ravel = pred.ravel()

    # Find index 255. or 1. region
    true_ind = np.where(true_ravel == 1)
    pred_ind = np.where(pred_ravel == 255)

    # Intersection , Union , Diff Calculation
    TP_ind = np.intersect1d(true_ind, pred_ind)
    FN_ind = np.setdiff1d(true_ind, TP_ind)
    FP_ind = np.setdiff1d(pred_ind,TP_ind)
    union_ind = reduce(np.union1d,(TP_ind, FN_ind, FP_ind))

    # Intersection of Union(HED,GT)


    TP_count = TP_ind.shape[0]
    union_count=union_ind.shape[0]
    pred_count = pred_ind[0].shape[0]
    true_count = true_ind[0].shape[0]

    precision = 0
    iou = 0
    recall =0
    f1 = 0
    print('THRES({}) - TP : {}, UNION : {}, PRED : {}, TRUE : {}'.format(threshold, TP_count, union_count,pred_count, true_count))
    if TP_count==0 or pred_count==0 or true_count==0 or union_count==0:
        pass

    else :
        iou= TP_count / union_count
        precision = TP_count / pred_count
        recall = TP_count / true_count
        print(precision,recall)

        f1 = 2 * (precision * recall) / (precision + recall)

    # Create dummy array
    union = np.zeros(nflat)
    TP = np.zeros(nflat)
    FN = np.zeros(nflat)
    FP = np.zeros(nflat)

    # Write Array
    union[union_ind]=255
    TP[TP_ind]=255
    FN[FN_ind]=255
    FP[FP_ind]=255

    # return 2d arrays and iou
    return np.reshape(union,true.shape), np.reshape(TP,true.shape),np.reshape(FP,true.shape),np.reshape(FN,true.shape),precision,recall,iou ,f1

def plotResults2(inp, images, gt, size,fname,thres):
    images[0] = images[0].astype(int)

    images[0][images[0] < thres] = 0
    images[0][images[0] >= thres] = 255
    pylab.rcParams['figure.figsize'] = size, size
    gts=2
    if evaluation == False :
        gts=1
    nPlots = len(images)+2
    titles = ['INPUT','GT','HED', 'S1', 'S2', 'S3', 'S4']
    for i in range(0,gts):
        s = plt.subplot(2,5,i+1)
        if i ==0:
            plt.imshow(inp,cmap=cm.Greys_r)
            #plt.imshow(inp)
        if i==1:
            plt.imshow(gt,cmap=cm.Greys_r)
            #plt.imshow(gt)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
        s.set_title(titles[i], fontsize=35)
    for i in range(0, len(images)):
        s = plt.subplot(2, 5, i + 6)
        plt.imshow(images[i], cmap=cm.Greys_r)
        #plt.imshow(images[i])
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
        s.set_title(titles[i+2], fontsize=35)
    plt.tight_layout()

    # if 'croppedimg' in fname[0].split('/'):
    if 'filteredimg' in fname[0].split('/'):
        plt.savefig(os.path.join('output',fname[0][16:].replace('jpg','jpeg')))
    else:
        plt.savefig(os.path.join('output',fname[0][16:].replace('jpg','jpeg')))

def plotResults_without_gt(inp, images, gt, size,fname,thres):
    images[0] = images[0].astype(int)
    print(images[0])
    images[0][images[0] < thres] = 0
    images[0][images[0] >= thres] = 255
    pylab.rcParams['figure.figsize'] = size, size
    gts=2
    plt.axis('off')
    if evaluation == False :
        gts=1
    for i in range(0,gts):
        if i ==0:
            plt.imshow(inp)
        if i==1:
            plt.imshow(gt,cmap=cm.Greys_r)

    img_contour = images[0].astype(np.uint8)
    contours, hierachy = cv2.findContours(img_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area_list = []
    area_ind = []
    k=0
    for j in range(len(contours)):
        area_list.append(int(cv2.contourArea(contours[j])))
        if len(contours[j]) > 1:
            x0, y0 = zip(*np.squeeze(contours[j]))
            k +=1
            plt.plot(x0, y0, c="b", linewidth=1.0)
            area_ind.append(k)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join('output',fname[0] + '_contour.png'), bbox_inches='tight', dpi=400, pad_inches=0)

    h = inp.shape[0]
    w = inp.shape[1]
    zeros = np.zeros((h, w))
    y_pred = images[0]
    ones = y_pred.reshape(h, w)
    mask = np.stack((ones, zeros, zeros, ones), axis=-1)

    plt.imshow(mask, alpha=0.3)
    plt.axis('off')
    plt.savefig(os.path.join('output',fname[0] + '_maskContour.png'), dpi=400, pad_inches=0,
                bbox_inches='tight')
    plt.clf()

    plt.imshow(inp)
    plt.imshow(mask, alpha=0.3)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join('output',fname[0] + '_mask_overlay.png'), pad_inches=0, bbox_inches='tight',
                dpi=400)
    plt.clf()

def plotResults(inp, images, gt, size,fname,thres):
    pylab.rcParams['figure.figsize'] = size, size
    gts=4
    loss_is = 3

    prev_union, prev_TP, prev_FP, prev_FN, prev_precision, prev_recall, prev_iou, prev_f1 = union_intersect(gt, images[0], threshold=thres-20)
    union, TP, FP, FN, precision, recall, iou, f1 = union_intersect(gt,images[0],threshold=thres)
    next_union, next_TP, next_FP, next_FN, next_precision, next_recall, next_iou, next_f1 = union_intersect(gt, images[0], threshold=thres+20)

    
    
    if evaluation == False :
        gts=1
        loss_is=0

    for i in range(0,loss_is):
        s = plt.subplot(3,5,i+1)
        if i==0:
            title= '{} F1 : {}'.format(thres-20,round(prev_f1,3))
            plt.imshow(np.dstack((prev_TP+prev_FP,prev_FN+prev_FP,np.zeros(TP.shape))))
        if i==1:
            title = '{} F1 : {}'.format(thres,round(f1, 3))
            plt.imshow(np.dstack((TP+FP, FN+FP, np.zeros(TP.shape))))
            img = np.dstack((TP+FP, FN+FP, np.zeros(TP.shape)))
            b,g,r = cv2.split(img)
            img2 = cv2.merge([r,g,b])
            y_pred = TP+FP

            
            h = inp.shape[0]
            w = inp.shape[1]
            zeros = np.zeros((h, w))
            y_pred = TP+FP
            ones = y_pred.reshape(h, w)
            mask = np.stack((ones, zeros, zeros, ones), axis=-1)

            inp_reshape = Image.open(arg_DataRoot+'\\' + fname[0]).convert('RGBA')
            img3 = inp_reshape + mask
            b2,g2,r2,a2 = cv2.split(img3)
            img4 = cv2.merge([r2,g2,b2,a2])
            cv2.imwrite(os.path.join('output',fname[0].split('croppedimg')[-1][1:]), img2)
            cv2.imwrite(os.path.join('output',fname[0].split('croppedimg')[-1][1:]) + '.png', TP+FP)
            cv2.imwrite(os.path.join('output',fname[0].split('croppedimg')[-1][1:]) + '_overlay.jpg', img4)
            
        if i==2:
            title = '{} F1 : {}'.format(thres+20,round(next_f1, 3))
            plt.imshow(np.dstack((next_TP+next_FP, next_FN+next_FP, np.zeros(TP.shape))))

        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
        s.set_title(title, fontsize=35)

    titles = ['INPUT', 'GT', 'UNION(GT,HED)', 'INTER(GT,HED)', 'HED', 'S1', 'S2', 'S3', 'S4']
    for i in range(0,gts):
        s = plt.subplot(3,5,i+6)
        if i ==0:
            plt.imshow(inp,cmap=cm.Greys_r)          
            #plt.imshow(inp)
        if i==1:
            plt.imshow(gt.astype(np.uint8), cmap=cm.Greys_r)
            #plt.imshow(gt)
        if i==2:
            plt.imshow(union,cmap=cm.Greys_r)
            #plt.imshow(union)
        if i==3:
            plt.imshow(TP,cmap=cm.Greys_r)
            #plt.imshow(TP)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
        s.set_title(titles[i], fontsize=35)
       

    for i in range(0, len(images)):
        s = plt.subplot(3, 5, i + 11)
        plt.imshow(images[i], cmap=cm.Greys_r)
        #plt.imshow(images[i])
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
        s.set_title(titles[i+4], fontsize=35)
    plt.tight_layout()

    if 'croppedimg' in fname[0]:
        plt.savefig(os.path.join('output',fname[0].split('croppedimg')[-1][1:].replace('jpg','jpeg')))
    else:
        #plt.savefig(os.path.join('output',fname[0].split('/')[2].replace('jpg','jpeg')))
        plt.savefig(os.path.join('output',fname[0].split('croppedimg')[-1][1:].replace('jpg','jpeg')))

    return fname[0], thres, iou, precision, recall, f1

def plot_contour_overlay(inp, images, gt, size,fname,thres):

    union, TP, FP, FN, precision, recall, iou, f1 = union_intersect(gt,images[0],threshold=thres)
    
    y_pred = TP+FP
    
    img_contour =y_pred.astype(np.uint8)
    contours, hierachy = cv2.findContours(img_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    area_list = []
    for i in range(len(contours)):
        if len(contours[i]) > 1 and cv2.contourArea(contours[i]) > 200:
            area_list.append(cv2.contourArea(contours[i]))
            x0, y0 = zip(*np.squeeze(contours[i]))
            plt.plot(x0, y0, c="b", linewidth = 1.0)
       
    h = inp.shape[0]
    w = inp.shape[1]
    zeros = np.zeros((h, w))
    y_pred = TP+FP
    ones = y_pred.reshape(h, w)
    mask = np.stack((ones, zeros, zeros, ones), axis=-1)
    plt.imshow(inp)
    plt.imshow(mask, alpha=0.3)
    plt.xlim(0, w)
    plt.ylim(h, 0)
    
    plt.savefig(os.path.join('output',fname[0][16:-4]) + '_contour_overlay.jpg', dpi = 300)
    plt.clf()
    return fname[0], thres, iou, precision, recall, f1
    

if __name__ == '__main__':

    nVisualize = 1
    inp = None
    fname = None
    gt = None
    input_img = None
    gt_img = None

    # arg_Model : which model to use
    # arg_DataRoot : path to the dataRoot
    # arg_thres : threshold of the image output from the model

    arg_Model = 'train/HED0_v3.pth'
    arg_DataRoot = 'data/dam_material_falloff/'
    arg_Thres = 160
    filter_bool = False
    fil_thresh = 180
    fil_kernel = 3
    option = ''

    for Opt, Arg in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
        if Opt == '--model' and Arg != '': arg_Model = Arg
        if Opt == '--data' and Arg != '': arg_DataRoot = Arg
        if Opt == '--thres' and Arg != '': arg_Thres = float(Arg)

    # using evaluation metrics(Must have data in the croppedgt directory)
    evaluation = True



    # fix random seed
    rng = np.random.RandomState(37148)

    # create instance of HED model
    net = HED()
    net.cuda()


    # load the weights for the model
    net.load_state_dict(torch.load(arg_Model))

    # batch size
    nBatch = 1

    # make test list for infer
    make_txt(arg_DataRoot,'test')

    # create data loaders from dataset
    # testPath = os.path.join(arg_DataRoot, 'test.lst') # linux version
    testPath = arg_DataRoot + '/' + 'test.lst'

    # create data loaders from dataset
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    # std=[0.229, 0.224, 0.225]
    # mean=[0.185, 0.156, 0.106]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    targetTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    testDataset = None

    if evaluation:
        if filter_bool:
            filter_thresh = fil_thresh
            filter_kernel = fil_kernel
            testDataset = TestDataset(testPath, arg_DataRoot, filter_kernel, filter_thresh, transform, targetTransform)
            filter_class = 'median'
        else:
            filter_thresh = 0
            filter_kernel = 0
            testDataset = TestDatasetNonFilter(testPath, arg_DataRoot, transform, targetTransform)
            filter_class = 'None'
    else:
        if filter_bool:
            filter_thresh = fil_thresh
            filter_kernel = fil_kernel
            testDataset = TestDataset(testPath, arg_DataRoot, filter_kernel, filter_thresh, transform)
            filter_class = 'median'
        else:
            filter_thresh = 0
            filter_kernel = 0
            testDataset = TestDatasetNonFilter(testPath, arg_DataRoot, transform, targetTransform)
            filter_class = 'None'
    testDataloader = DataLoader(testDataset, batch_size=nBatch)

    if os.path.exists('output') == False:
        os.mkdir('output')
    if os.path.exists('output/pred') == False:
        os.mkdir('output/pred')

    f = open(os.path.join('output', 'output.csv'), 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(['inference', 'data', 'infer_thres', 'Normalization', 'preprocessing(filter)', 'kernel', 'kernel threshold',
                 'inference threshold', 'option'])
    wr.writerow(['', arg_DataRoot, arg_Thres, [mean, std], filter_class, filter_kernel, filter_thresh,
                 arg_Thres, option])
    wr.writerow(['filename', 'IoU', 'Precision', 'Recall', 'F1 score'])
    for i, sample in enumerate(testDataloader):
        # get input sample image
        if evaluation :
            inp, fname, gt = sample
            gt = Variable(gt)
            file_name=fname[0]
        else :
            inp, fname = sample
        inp = Variable(inp.cuda())

        iou,precision,recall,f1 = 0.0, 0.0, 0.0, 0.0
        # perform forward computation
        s1, s2, s3, s4, s5, s6 = net.forward(inp)

        # convert back to numpy arrays

        out = []
        out.append(grayTrans(s6.data.cpu()))
        out.append(grayTrans(s1.data.cpu()))
        out.append(grayTrans(s2.data.cpu()))
        out.append(grayTrans(s3.data.cpu()))
        out.append(grayTrans(s4.data.cpu()))

        #out.append(s6.data.cpu())
        #out.append(s1.data.cpu())
        #out.append(s2.data.cpu())
        #out.append(s3.data.cpu())
        #out.append(s4.data.cpu())

        # inp2 = inp.data[0].permute(1, 2, 0)

        # input_img = inp2.cpu().numpy()
        input_img = inp.data[0].cpu().numpy()[0]
        if evaluation:
            print(fname[0])
            img = Image.fromarray(out[0], 'L')
            # img.save(os.path.join('output/pred', fname[0].split('/')[2].replace('jpg', 'jpeg')))
            img.save(os.path.join('output/pred', fname[0].split('croppedimg')[-1][1:].replace('jpg', 'jpeg')))
            print(fname[0].split('croppedimg')[-1][1:].replace('jpg', 'jpeg'))
            gt_img = gt.data[0].cpu().numpy()[0]
            print(gt_img.shape)

        # visualize every 10th image
        if i % nVisualize == 0:
            # if (len(gt_unique[0])==2 and gt_unique[1][1]>1000): # unique value is not 0.
            #     print(len(gt_unique[0]),gt_unique[1][1])
                if evaluation:
                    file_name, thres, iou, precision, recall, f1=plotResults(input_img, out, gt_img, 25, fname, arg_Thres)
                    # file_name, thres, iou, precision, recall, f1=plot_contour_overlay(input_img, out, gt_img, 25, fname, arg_Thres)
                    f = open(os.path.join('output', 'output.csv'), 'a', encoding='utf-8', newline='')
                    file_name = file_name.split('croppedimg')[-1][1:]
                    print(file_name)
                    # file_name = file_name.split('filteredimg')[-1][1:]
                    wr = csv.writer(f)
                    wr.writerow([file_name, iou, precision, recall, f1])
                    f.close()

                else:
                    # plotResults2(input_img,out,gt_img, 25,fname,arg_Thres)
                    plotResults_without_gt(input_img,out,gt_img, 25,fname,arg_Thres)
