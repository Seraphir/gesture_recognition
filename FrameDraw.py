import os
import cv2
import numpy as np
import shutil

input_dir = ".\\data\\cropgest"
output_dir = ".\\data\\dcgest"
data_type = "_oth"
data_dir1 = os.path.join(input_dir, "atmDataSet{}".format(data_type))
data_dir2 = os.path.join(input_dir, "RdDataSet{}".format(data_type))
store_dir1 = os.path.join(output_dir, "atmDataSet{}".format(data_type))
store_dir2 = os.path.join(output_dir, "RdDataSet{}".format(data_type))


def rand_start_stop():
    blank = np.abs(np.random.randn(2)) * 6
    blank[blank > 8] = 8
    start = blank[0]
    stop = 31 - blank[1]
    reserve = np.floor(np.linspace(start, stop, 16)).astype(np.int32)
    start = int((start / 32) * 112)
    stop = int((stop / 32) * 112)
    return start, stop, reserve


if not os.path.exists(store_dir1):
    os.makedirs(store_dir1)
if not os.path.exists(store_dir2):
    os.makedirs(store_dir2)
total_cnt = []
for classname in os.listdir(data_dir1):
    class_cnt = 0
    sub_data_dir1 = os.path.join(data_dir1, classname)
    sub_data_dir2 = os.path.join(data_dir2, classname)
    sub_store_dir1 = os.path.join(store_dir1, classname)
    sub_store_dir2 = os.path.join(store_dir2, classname)
    if not os.path.exists(sub_store_dir1):
        os.makedirs(sub_store_dir1)
    if not os.path.exists(sub_store_dir2):
        os.makedirs(sub_store_dir2)
    for idx in os.listdir(sub_data_dir2):
        class_cnt += 1
        src_path1 = os.path.join(sub_data_dir1, "{}.png".format(idx))
        dst_path1 = os.path.join(sub_store_dir1, "{}.png".format(idx))
        atm_mat = cv2.imread(src_path1)
        x0, x1, frame_ls = rand_start_stop()
        atm_mat = cv2.resize(atm_mat[:, x0:x1], (112, 112))
        cv2.imwrite(dst_path1, atm_mat)
        sub_sub_store_dir2 = os.path.join(sub_store_dir2, idx)
        if not os.path.exists(sub_sub_store_dir2):
            os.makedirs(sub_sub_store_dir2)
        for j, resever_frame in enumerate(frame_ls):
            src_path2 = os.path.join(sub_data_dir2, idx, "{:02d}.png".format(resever_frame))
            dst_path2 = os.path.join(sub_sub_store_dir2, "{:02d}.png".format(j))
            shutil.copy(src_path2, dst_path2)
        print(classname, idx)
    total_cnt.append(class_cnt)
print(total_cnt)
