import os
import cv2
import numpy as np
import pickle as pkl
import scipy.io as io

root_dir = ".\\data\\target"
data_type = ""
subdata_dir1 = os.path.join(root_dir, "atmDataSet{}".format(data_type))
subdata_dir2 = os.path.join(root_dir, "RDDataSet{}".format(data_type))
storedata_dir = os.path.join(root_dir, "ardtmDataSet{}".format(data_type))

if not os.path.exists(storedata_dir):
    os.makedirs(storedata_dir)

total_cnt = []
for classname in os.listdir(subdata_dir1):
    class_cnt = 0
    sub_dir1 = os.path.join(subdata_dir1, classname)
    sub_dir2 = os.path.join(subdata_dir2, classname)
    sub_dir = os.path.join(storedata_dir, classname)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    for idx in os.listdir(sub_dir2):
        class_cnt += 1
        save_path = os.path.join(sub_dir, "{}.png".format(idx))
        atm_mat = cv2.imread(os.path.join(sub_dir1, "{}.png".format(idx)))
        if atm_mat.ndim > 2:
            atm_mat = atm_mat[:, :, 0]
        img_dir = os.path.join(sub_dir2, "{}".format(idx), "00.png")
        rd_mat = cv2.imread(img_dir)
        h0, w0 = rd_mat.shape[:2]
        rtm_mat = np.zeros((w0, 32))
        dtm_mat = np.zeros((h0, 32))
        for j in range(32):
            img_dir = os.path.join(sub_dir2, "{}".format(idx), "{:02d}.png".format(j))
            rd_mat = cv2.imread(img_dir)
            rd_mat = rd_mat[:, :, 0] if rd_mat.ndim > 2 else rd_mat
            y, x = np.where(rd_mat == np.max(rd_mat))
            y = y[0] if len(y) > 1 else y
            x = x[0] if len(x) > 1 else x
            rtm_mat[:, j] = rd_mat[y, :]
            dtm_mat[:, j] = rd_mat[:, x].T
        h, w = atm_mat.shape
        rtm_mat = cv2.resize(rtm_mat, (h, w))
        dtm_mat = cv2.resize(dtm_mat, (h, w))
        ardtm_mat = np.vstack([np.hstack([atm_mat, rtm_mat]), np.hstack([dtm_mat, atm_mat])])
        ardtm_mat = cv2.resize(ardtm_mat, (h, w))
        cv2.imwrite(save_path, ardtm_mat)
        print(classname, idx)
    total_cnt.append(class_cnt)
print(total_cnt)
