import os
import cv2
import numpy as np
import pickle as pkl
import scipy.io as io
import matplotlib.pyplot as plt

root_dir = ".\\data\\cropgest"

subdata_dir1 = os.path.join(root_dir, ".\\atmMat")
subdata_dir2 = os.path.join(root_dir, ".\\RdMat")
storedata_dir1 = os.path.join(root_dir, ".\\atmDataSet_aug")
storedata_dir2 = os.path.join(root_dir, ".\\RdDataSet_aug")
class_list = os.listdir(subdata_dir1)
check_list = os.listdir(subdata_dir2)

atm_mat_dict = {}
rd_mats_dict = {}
try:
    atm_mat_dict = pkl.load(open(os.path.join(root_dir, "atm.pkl"), 'rb'))
    rd_mats_dict = pkl.load(open(os.path.join(root_dir, "rd.pkl"), 'rb'))
except:
    for i, class_name in enumerate(class_list):
        if class_name in check_list:
            atm_names = os.listdir(os.path.join(subdata_dir1, class_name))
            rd_names = os.listdir(os.path.join(subdata_dir2, class_name))
            atm_class_ls = []
            rd_class_ls = []
            for j, atm_name in enumerate(atm_names):
                if atm_name.split('.')[0] == rd_names[j]:
                    atm_dir = os.path.join(subdata_dir1, class_name, atm_name)
                    atm_mat = io.loadmat(atm_dir)["atm"]
                    atm_class_ls.append(atm_mat)

                    rd_dir = os.path.join(subdata_dir2, class_name, rd_names[j])
                    rd_files = os.listdir(rd_dir)
                    rd_mats = [io.loadmat(os.path.join(rd_dir, rd_file))["rd"] for rd_file in rd_files]
                    rd_class_ls.append(rd_mats)
                    print(i, j)
            atm_mat_dict[class_name] = atm_class_ls
            rd_mats_dict[class_name] = rd_class_ls
    pkl.dump(atm_mat_dict, open(os.path.join(root_dir, "atm.pkl"), "wb"))
    print("atm saved")
    pkl.dump(rd_mats_dict, open(os.path.join(root_dir, "rd.pkl"), "wb"))
    print("rd saved")


class data_augmentation:

    def __init__(self, dist_t=0, scale_t=0., gray_t=0.):
        self.dist_t = dist_t
        self.scale_t = scale_t
        self.gray_t = gray_t

        # transform random param
        self.dist_mv = 0
        self.scale_fx = 1
        self.gray_idx = 1

    def transform(self, mat):
        mat = self._dist_transform(mat)
        mat = self._scale_transform(mat)
        mat = self._gray_transform(mat)
        return mat

    def rand_param(self):
        self.dist_mv = self.dist_t * np.random.randn()
        self.scale_fx = self.scale_t * np.random.randn() + 1
        self.gray_idx = self.gray_t * np.random.randn() + 1

        # check
        if self.dist_mv < - self.dist_t:
            self.dist_mv = - self.dist_t
        elif self.dist_mv > self.dist_t:
            self.dist_mv = self.dist_t

        if self.scale_fx < 1 - self.scale_t:
            self.scale_fx = 1 - self.scale_t
        elif self.scale_fx > 1 + self.scale_t:
            self.scale_fx = 1 + self.scale_t

        if self.gray_idx < 1 - self.gray_t:
            self.gray_idx = 1 - self.gray_t
        elif self.gray_idx > 1 + self.gray_t:
            self.gray_idx = 1 + self.gray_t

    def _dist_transform(self, mat):
        rows, cols = mat.shape[:2]
        m = np.float32([[1, 0, self.dist_mv], [0, 1, 0]])
        mat = cv2.warpAffine(mat, m, (rows, cols))
        return mat

    def _scale_transform(self, mat):
        rows, cols = mat.shape[:2]
        m = np.float32([[self.scale_fx, 0, 0], [0, self.scale_fx, int((1 - self.scale_fx) * rows / 2)]])
        mat = cv2.warpAffine(mat, m, (rows, cols))
        return mat

    def _gray_transform(self, mat):
        mat = mat ** self.gray_idx
        return mat


# if not os.path.exists(storedata_dir1):
#     os.makedirs(storedata_dir1)
# for classname, atm_class_ls in atm_mat_dict.items():
#     sub_dir = os.path.join(storedata_dir1, classname)
#     if not os.path.exists(sub_dir):
#         os.makedirs(sub_dir)
#     for i, atm_mat in enumerate(atm_class_ls):
#         save_path = os.path.join(sub_dir, "{:04d}.png".format(i))
#         atm_mat = cv2.resize(atm_mat, (224, 224))
#         cv2.imwrite(save_path, np.uint8(atm_mat * 255))
#         print(classname, i)
# cv2.imshow("atm", atm_mat)
# cv2.waitKey(0)

# element = np.vstack([np.zeros(63), np.linspace(0, 1, 63), np.zeros(63)])
# test_img = np.tile(element, (21, 1)).astype(np.float64)
# test_img[31, :] = np.flip(np.linspace(0, 1, 63))
base_times = 3
if not os.path.exists(storedata_dir1):
    os.makedirs(storedata_dir1)
if not os.path.exists(storedata_dir2):
    os.makedirs(storedata_dir2)
DataAug = data_augmentation(dist_t=3, scale_t=0.2, gray_t=0.2)
for base_times in range(4):
    total_cnt = []
    for classname, rd_class_ls in rd_mats_dict.items():
        class_cnt = 0
        # if classname not in ["youzuohua", "zuoyouhua"]:
        #     continue
        class_num = len(rd_class_ls)
        sub_dir1 = os.path.join(storedata_dir1, classname)
        sub_dir2 = os.path.join(storedata_dir2, classname)
        if not os.path.exists(sub_dir1):
            os.makedirs(sub_dir1)
        for i, rd_mats in enumerate(rd_class_ls):
            class_cnt += 1
            save_path1 = os.path.join(sub_dir1, "{:04d}.png".format(i + base_times * class_num))
            atm_mat = atm_mat_dict[classname][i]
            atm_mat = cv2.resize(atm_mat, (224, 224))
            cv2.imwrite(save_path1, np.uint8(atm_mat * 255))

            sub_sub_dir2 = os.path.join(sub_dir2, "{:04d}".format(i + base_times * class_num))
            if not os.path.exists(sub_sub_dir2):
                os.makedirs(sub_sub_dir2)
            max_value = max([np.max(rd_mat) for rd_mat in rd_mats])
            DataAug.rand_param()
            for j, rd_mat in enumerate(rd_mats):
                save_path2 = os.path.join(sub_sub_dir2, "{:02d}.png".format(j))
                # rd_mat = (rd_mat / max_value) ** 0.4
                rd_mat = rd_mat / max(np.max(rd_mat), 1)
                rd_mat_t = DataAug.transform(rd_mat)
                cv2.imwrite(save_path2, np.uint8(rd_mat * 255))

                # test
                # test_img = np.hstack([test_img[:, 5:], test_img[:, :5]])
                # test_img_t = DataAug.transform(test_img)
                # show = np.vstack([test_img, test_img_t])

                # show = np.hstack([rd_mat, rd_mat_t])
                # cv2.imshow("atm", show)
                # cv2.waitKey(0)
            print(classname, i)
        total_cnt.append(class_cnt)
    print(total_cnt)
