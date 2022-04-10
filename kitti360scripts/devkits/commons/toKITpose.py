# -*- coding:utf-8 -*-
"""
读取kitti360 的cam0_to_world.txt poses.txt 化为正常的kitti轨迹文件
"""
import os
import argparse
from typing import Sequence
import numpy as np
import copy
import matplotlib.pyplot as plt
from loadCalibration import loadCalibrationCameraToPose
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 23,
}
#/media/poss/dataset/poses/00.txt /media/poss/r15posestart0.txt
poesroot = '/media/KITTI-360/dataset/poses'
dataroot = '/media/KITTI-360/dataset'

#读入位姿 读取KITTI格式的轨迹 返回序列的变换矩阵 
def loadpose(filename):
    file = open(filename)
    file_lines = file.readlines()
    numberOfLines = len(file_lines)
    dataArray = np.zeros((numberOfLines, 4, 4))
    index = 0 #一行 17 个 舍弃第一个数
    frameid=[]
    for line in file_lines:
        line = line.strip() # 参数为空时，默认删除开头、结尾处空白符（包括'\n', '\r',  '\t',  ' ')
        formLine = line.split( ) #按空格切分？
        dataArray[index] = np.eye(4) 
        dataArray[index, 0, 0:4] = formLine[1:5]
        dataArray[index, 1, 0:4] = formLine[5:9]
        dataArray[index, 2, 0:4] = formLine[9:13]
        index += 1
        frameid.append(int(formLine[0]))
    file.close()
    # #需要对位姿 对齐首帧
    # Tw0 = copy.deepcopy(dataArray[0])
    # T0w = np.linalg.inv(Tw0)
    # for i in range(dataArray.shape[0]):
    #     oldTwi = copy.deepcopy(dataArray[i])
    #     newT0i = np.matmul(T0w,oldTwi)
    #     dataArray[i] = newT0i
    frameid = np.array(frameid)
    return dataArray,frameid # n,4,4

def qieseg(startf,endf): #cam0_to_world poses 00/cam0_to_world.txt
    gtpath = os.path.join(poesroot,'cam0_to_world_8.txt')
    rawgt, frameid = loadpose(gtpath)
    rawlen = rawgt.shape[0]
    print('load pose: ', gtpath)
    print('raw lenth: ',rawlen)
    if endf<0:
        endf = rawlen-1
    segpose = rawgt[startf:endf+1]
    len = segpose.shape[0]
    outpose = np.zeros((len, 4, 4))
    Tw0 = copy.deepcopy(segpose[0])
    T0w = np.linalg.inv(Tw0)
    outpath = os.path.join(poesroot,'00_8.txt')#'system 00_cam0_2
    fp = open(outpath,'w')
    for i in range(len):
        Twi = segpose[i]
        T0i = np.matmul(T0w,Twi) #np.matmul(T0w,Twi) Twi
        outpose[i] = T0i
        fp.write('{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n'
                 .format(T0i[0,0],T0i[0,1],T0i[0,2],T0i[0,3],
                         T0i[1,0],T0i[1,1],T0i[1,2],T0i[1,3],
                         T0i[2,0],T0i[2,1],T0i[2,2],T0i[2,3] )
                )
    
    # 差分 得到 framid 间隔
    dfid = frameid[1:]-frameid[:-1]
    dfnum = dfid.shape[0]
    print('dfid size: ', dfnum)
    fx = np.linspace(0,dfnum,dfnum)
    # 画图 看看
    fig1, axs1 = plt.subplots(figsize=(20, 10)) #画值
    axs1.plot(fx,dfid)
    fig1.tight_layout()
    plt.show()

def qierawseg(startf,endf):
    rawgt = np.loadtxt(os.path.join(poesroot,'00/cam0_to_world.txt'))
    rawlen = rawgt.shape[0]
    if endf<0:
        endf = rawlen-1
    segpose = rawgt[startf:endf+1,:]
    frameid = segpose[:,0]
    len = segpose.shape[0]

    # outpath = os.path.join(poesroot,'cam0_to_world_7.txt')#'system
    # np.savetxt(outpath,segpose,fmt='%d %9f %9f %9f %9f %9f %9f %9f %9f %9f %9f %9f %9f %9f %9f %9f %9f')
    # 差分 得到 framid 间隔
    dfid = frameid[1:]-frameid[:-1]
    dfnum = dfid.shape[0]
    print('dfid size: ', dfnum)
    fx = np.linspace(0,dfnum,dfnum)
    # 画图 看看
    fig1, axs1 = plt.subplots(figsize=(20, 10)) #画值
    axs1.plot(fx,dfid)
    fig1.tight_layout()
    fig2, axs2 = plt.subplots(figsize=(20, 10)) #画值
    fx = np.linspace(0,frameid.shape[0],frameid.shape[0])
    axs2.plot(fx,frameid)
    fig2.tight_layout()
    plt.show()

def main(seq):
    camfile = os.path.join(poesroot,seq,'cam0_to_world.txt') #作者说 这是 rect0系
    systemfile = os.path.join(poesroot,seq,'poses.txt')
    cam, frameidcam = loadpose(camfile)
    system, frameidsys = loadpose(systemfile)
    rawlen = system.shape[0]
    outassfile = os.path.join(dataroot,'sequences',seq,'association.txt')
    np.savetxt(outassfile,frameidsys,fmt='%d') # 保存gt存在的数据序号

    # 差分 得到 framid 间隔
    dfids = frameidsys[1:]-frameidsys[:-1]
    dfnums = dfids.shape[0]
    print('dfids size: ', dfnums)
    fx = np.linspace(0,dfnums,dfnums)
    # 画图 看看
    fig1, axs1 = plt.subplots(figsize=(20, 10)) #画值
    axs1.plot(fx,dfids)
    fig1.tight_layout()
    dfidc = frameidcam[1:]-frameidcam[:-1]
    dfnumc = frameidcam.shape[0]
    print('pose size: ', dfnumc)
    fx = np.linspace(0,dfnumc,dfnumc)
    fig2, axs2 = plt.subplots(figsize=(20, 10)) #画值
    axs2.plot(fx,frameidcam)
    fig2.tight_layout()
    plt.show()
    
    Rrect = np.array([[ 9.99974e-01, -7.14100e-03, -8.90000e-05,  0.00000e+00],
                     [ 7.14100e-03,  9.99969e-01, -3.24700e-03,  0.00000e+00],
                     [ 1.12000e-04,  3.24700e-03,  9.99995e-01,  0.00000e+00],
                     [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]])
    print('Rrect: \n',Rrect)
    Rrectinv = np.linalg.inv(Rrect) #T_CR

    fileCameraToPose = os.path.join(dataroot, 'calib', 'calib_cam_to_pose.txt')#作者说 这是相机系 不是rectifid
    Tr = loadCalibrationCameraToPose(fileCameraToPose)
    T_BC = Tr['image_00']
    T_BR = np.matmul(T_BC,Rrectinv)# rect0到车体的变换
    T_RB = np.linalg.inv(T_BR)
    T_CB = np.linalg.inv(T_BC)
    print('Loaded %s' % fileCameraToPose)
    print('T_CB: \n',T_CB)
    print('T_BC*T_CB: \n',np.matmul(T_BC,T_CB))
    outpath1 = os.path.join(poesroot,seq,'kitpose2rect0.txt')#'kitpose2cam.txt
    fpp = open(outpath1,'w')
    Tw0 = copy.deepcopy(system[0])
    T0w = np.linalg.inv(Tw0)
    for i in range(rawlen):
        Twi = system[i]
        T0i = np.matmul(T0w,Twi)
        T0i2c =  np.matmul(np.matmul(T_RB,T0i),T_BR)
        fpp.write('{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n'
                 .format(T0i2c[0,0],T0i2c[0,1],T0i2c[0,2],T0i2c[0,3],
                         T0i2c[1,0],T0i2c[1,1],T0i2c[1,2],T0i2c[1,3],
                         T0i2c[2,0],T0i2c[2,1],T0i2c[2,2],T0i2c[2,3] )
                )


    outsys = np.zeros((rawlen, 4, 4))
    outpath = os.path.join(poesroot,seq,'kitposes.txt')#'system 00_cam0_2
    fp = open(outpath,'w')
    for i in range(rawlen):
        Twi = system[i]
        T0i = np.matmul(T0w,Twi) #np.matmul(T0w,Twi) Twi
        outsys[i] = T0i
        fp.write('{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n'
                 .format(T0i[0,0],T0i[0,1],T0i[0,2],T0i[0,3],
                         T0i[1,0],T0i[1,1],T0i[1,2],T0i[1,3],
                         T0i[2,0],T0i[2,1],T0i[2,2],T0i[2,3] )
                )
    
    outcam = np.zeros((cam.shape[0], 4, 4))
    Tw0c = copy.deepcopy(cam[0])
    T0wc = np.linalg.inv(Tw0c)
    # outcpath = os.path.join(poesroot,seq,'kitrect0.txt')#'system 00_cam0_2 kitcam0
    # fpc = open(outcpath,'w')
    out2path = os.path.join(poesroot,seq,'kitcam0.txt')
    fp2 = open(out2path,'w')
    for i in range(cam.shape[0]):
        Twic = cam[i]
        T0ic = np.matmul(T0wc,Twic) #np.matmul(T0w,Twi) Twi
        # T0ir = np.matmul(np.matmul(Rrect,T0ic),Rrectinv) # cam0 本身就是rectified 不用再生成kitrect0.txt
        # fpc.write('{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n'
        #          .format(T0ir[0,0],T0ir[0,1],T0ir[0,2],T0ir[0,3],
        #                  T0ir[1,0],T0ir[1,1],T0ir[1,2],T0ir[1,3],
        #                  T0ir[2,0],T0ir[2,1],T0ir[2,2],T0ir[2,3] )
        #         )
        fp2.write('{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n'
                 .format(T0ic[0,0],T0ic[0,1],T0ic[0,2],T0ic[0,3],
                         T0ic[1,0],T0ic[1,1],T0ic[1,2],T0ic[1,3],
                         T0ic[2,0],T0ic[2,1],T0ic[2,2],T0ic[2,3] )
                )
    
    interfull = os.path.join(poesroot,seq,'kitrect0_full.txt')
    clipedfull = os.path.join(poesroot,seq,'kitrect0_full_less.txt')
    if not (os.path.exists(interfull)):
        return
    
    interpose = np.loadtxt(interfull)
    clippedpose = interpose[frameidcam] #再从插值后的 提出有gt的位姿
    print('clipp from full size: {:d}'.format(clippedpose.shape[0]))
    np.savetxt(clipedfull,clippedpose,fmt='%6f')

   
    
    # pass

if __name__ == '__main__':
    # parser command lines
    parser = argparse.ArgumentParser(description=''' ''') 
    # parser.add_argument('result_dir', help='预测的轨迹所在文件夹',default="/home/dlr/Project/Loc_fuse/result/lidarstereo/") #
    # parser.add_argument('startid', help='起始帧',default='0')
    # parser.add_argument('endid', help='结束帧（含）',default='0')
    parser.add_argument('seqid', help='序列',default='03')

    args = parser.parse_args()
    # 读取参数
    # estdir = args.result_dir
    seq = str(args.seqid)
    # print("\n report result...\n")
    main(seq)
    # startf = int(args.startid)
    # endf = int(args.endid)
    # qierawseg(startf,endf)