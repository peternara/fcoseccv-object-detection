import os
xmldir = '/media/e813/E/dataset/eccv/eccv/VisDrone2018-VID-val/xmlannotations'
# datasetdir = '/media/e813/E/dataset/eccv/eccv/VisDrone2018-VID-train'
# file = os.path.join(datasetdir,'index.txt')
# f = open(file,'w')
count=0
for seq in os.listdir(xmldir):
    seqpath = os.path.join(xmldir,seq)
    for n,xml_name in enumerate(os.listdir(seqpath)):
        count += 1
        if n%4==0:

            name = xml_name[:-4]
            # f.write('{} {}\n'.format(seq,name))
print(count)
# f.close()
# with open(file) as f:
#     xmls = f.readlines()
#     xmls =[x.strip("\n") for x in xmls]
#     xmls = [x.split(' ') for x in xmls]
# print(xmls[1:10])
