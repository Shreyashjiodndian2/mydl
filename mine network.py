from PIL import Image
from os import listdir
from os.path import join, isfile
def data_seperator(data_direc, Features_list, Labels_list):
    z = listdir(data_direc)
    for i in range(len(z)):
        Features_list.append(join(data_direc,str(z[i])))
        Labels_list.append(z[i].split(".")[0])
raw_Features=list()
Labels=list()
data_seperator("E:/datasets/train/train", Features_list=raw_Features, Labels_list=Labels)

img = Image.open(raw_Features[0]).convert("L")
img.load()

