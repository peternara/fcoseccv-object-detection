import sys
import os
import cv2
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
EccvPath = '/media/e813/E/dataset/eccv/eccv/'
DetPath = os.path.join(EccvPath,'VisDrone2018-DET-train')
AnnoPath = os.path.join(DetPath,'annotations')
ImgPath = os.path.join(DetPath,'images')
AnnoList = os.listdir(AnnoPath)
xmlDir = os.path.join(DetPath,'xmlannotations_k')
CLASSES = [
        "__background__",
        "pedestrian",
        "person",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor",
        "__background__",
    ]
num_dict = {}
for item in CLASSES:
    num_dict[item] = 0
# with open('/media/e813/E/dataset/eccv/eccv/VisDrone2018-DET-train/annotations/9999974_00000_d_0000053.txt') as f:
#     for line in f.readlines():
#         try:
#             objitem = [int(x) for x in line.split(',')]
#         except:
#             objitem = [int(x) for x in line[:-2].split(',')]
#             print(objitem)
# exit()
score_dict = {}
for anno in AnnoList:
    img_name = anno.split('.')[0]
    annopath = os.path.join(AnnoPath,img_name+'.txt')
    # imgpath = os.path.join(ImgPath,img_name+'.jpg')
    with open(annopath) as f:
        # root = ET.Element('annotation')
        # folder = ET.SubElement(root,'folder')
        # folder.text = 'VisDrone2018-DET-val'
        # filename = ET.SubElement(root, 'filename')
        # filename.text = '{}.jpg'.format(img_name)
        # img = cv2.imread(imgpath)
        # img_h,img_w,img_d=img.shape
        # size = ET.SubElement(root,'size')
        # width = ET.SubElement(size,'width')
        # width.text = str(img_w)
        # height = ET.SubElement(size, 'height')
        # height.text = str(img_h)
        # depth = ET.SubElement(size, 'depth')
        # depth.text = str(img_d)
        # segmented = ET.SubElement(root,'segmented')
        # segmented.text=str(0)
        for line in f.readlines():
            try:
                objitem = [int(x) for x in line.split(',')]
            except:
                objitem = [int(x) for x in line[:-2].split(',')]
            if objitem[4]!=0:
                num_dict[CLASSES[objitem[5]]]+=1
        #     if objitem[4] in score_dict:
        #         score_dict[objitem[4]]+=1
        #     else:
        #         score_dict[objitem[4]]=1
        #
        #     if objitem[5]==0:
        #         continue
        #     xmlobject = ET.SubElement(root,'object')
        #     objname = ET.SubElement(xmlobject,'name')
        #     objname.text=CLASSES[objitem[5]]
        #     truncation = ET.SubElement(xmlobject,'truncated')
        #     truncation.text=str(objitem[6])
        #     difficult = ET.SubElement(xmlobject,'difficult')
        #     if objitem[4]==0:
        #         difficult.text = str(1)
        #     else:
        #         difficult.text =str(0)
        #     bndbox = ET.SubElement(xmlobject,'bndbox')
        #     xmin = objitem[0]
        #     xmax = objitem[0]+objitem[2]
        #     ymin = objitem[1]
        #     ymax = objitem[1]+objitem[3]
        #     ET.SubElement(bndbox,'xmin').text=str(xmin)
        #     ET.SubElement(bndbox, 'xmax').text = str(xmax)
        #     ET.SubElement(bndbox, 'ymin').text = str(ymin)
        #     ET.SubElement(bndbox, 'ymax').text = str(ymax)
        # tree = ET.ElementTree(root)
        # xmlpath = os.path.join(xmlDir,img_name+'.xml')
        # print(xmlpath)
        # tree.write(xmlpath)
print(num_dict)

