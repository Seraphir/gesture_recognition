import os

root_dir = ".\\data\\cropgest"
index_list = ["index.txt", "index_50.txt", "index_oth.txt"]
outfile = open(os.path.join(root_dir, 'index_all.txt'), 'w+')
for index_file in index_list:
    infile = open(os.path.join(root_dir, index_file), 'r')
    msg = infile.read()
    outfile.write(msg)
    infile.close()
outfile.close()
