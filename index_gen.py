import os

subdata_dir1 = ".\\atmDataSet"
subdata_dir2 = ".\\RdDataSet"
class_list = os.listdir(subdata_dir1)
check_list = os.listdir(subdata_dir2)
f = open('.\\index.txt', 'w+')

for i, class_name in enumerate(class_list):
    if class_name in check_list:
        image_names = os.listdir(os.path.join(subdata_dir1, class_name))
        video_names = os.listdir(os.path.join(subdata_dir2, class_name))
        for j, image_name in enumerate(image_names):
            if image_name.split('.')[0] == video_names[j]:
                image_dir = os.path.join(subdata_dir1, class_name, image_name)
                video_dir = os.path.join(subdata_dir2, class_name, video_names[j])
                f.write("{} {} {}\n".format(image_dir, video_dir, i))
f.close()
