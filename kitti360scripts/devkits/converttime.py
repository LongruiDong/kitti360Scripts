# -*- coding:utf-8 -*-
"""
读取kitti360 的原始timestamps.txt 化为unix时间
"""

import os
import argparse
import numpy as np
import datetime
import matplotlib.pyplot as plt
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 23,
}
kt360root = '/media/KITTI-360/dataset/sequences'


def tontime(seq):
    timestamp_file = os.path.join(kt360root,seq,'timestamps.txt')
    print("raw time file: " + timestamp_file) 
    ntimes = [] 
    with open(timestamp_file, 'r') as f:
        for line in f.readlines():
            # NB: datetime only supports microseconds, but KITTI timestamps
            # give nanoseconds, so need to truncate last 4 characters to
            # get rid of \n (counts as 1) and extra 3 digits
            t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
            nt =t.timestamp() # 转为unix时间
            ntimes.append(nt)
    ntime = np.array(ntimes)
    ntime = ntime[:] - ntime[0]

    outpath = os.path.join(kt360root,seq,'times.txt')
    print("save to: " + outpath)
    np.savetxt(outpath, ntime, fmt= '%.6f') #保存时间戳

    # 画图 看看时间戳
    fig1, axs1 = plt.subplots(figsize=(20, 10)) #画值
    # 差分 得到 framid 间隔
    dfids = ntime[1:]-ntime[:-1]
    bad = np.where(np.abs(dfids)>0.12)
    print('bad time > 0.12: ',bad)
    dfnums = dfids.shape[0]
    # print('dfids size: ', dfnums)
    fx = np.linspace(0,dfnums,dfnums)
    axs1.plot(fx,dfids)
    fig1.tight_layout()
    plt.show()

if __name__ == '__main__':
    # parser command lines
    parser = argparse.ArgumentParser(description=''' ''') 
    parser.add_argument('seq', help='序列Id',default="00") # /timestamps.txt

    args = parser.parse_args()
    seq = str(args.seq)
    tontime(seq)