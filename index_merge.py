import os

root_dir = ".\\data\\cropgest"
# index_list = ["index.txt", "index_50.txt", "index_oth.txt"]
index_list = ["index2.txt", "index2_50.txt", "index2_oth.txt","index2_aug.txt"]
outfile = open(os.path.join(root_dir, 'index2_all_aug.txt'), 'w+')
for index_file in index_list:
    infile = open(os.path.join(root_dir, index_file), 'r')
    msg = infile.read()
    outfile.write(msg)
    infile.close()
outfile.close()
