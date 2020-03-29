import sys
import os
import cv2
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
EccvPath = '/media/e813/E/dataset/eccv/eccv/'
VidTrainPath = os.path.join(EccvPath,'VisDrone2018-VID-val')
AnnoPath = os.path.join(VidTrainPath,'annotations')
SeqPath = os.path.join(VidTrainPath,'sequences')
AnnoList = os.listdir(AnnoPath)
xmlDir = os.path.join(VidTrainPath,'xmlannotations')
CLASSES = [
        "__background__",
        "person",
        "person",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "tricycle",
        "bus",
        "motor",
        "__background__",
    ]

for anno in AnnoList:
    seq_name = anno[:-4]
    anno_path = os.path.join(AnnoPath,anno)
    seq_xml_dir = os.path.join(xmlDir,seq_name)
    if not os.path.exists(seq_xml_dir):
        os.mkdir(seq_xml_dir)
    seq_dict = {}
    with open(anno_path) as f:
        for line in f.readlines():
            item_line = [int(x) for x in line.split(',')]
            if item_line[0] not in seq_dict:
                seq_dict[item_line[0]] =[]
            seq_dict[item_line[0]].append(item_line[1:])
    for key in seq_dict:
        img_name = '%07d' % key
        img_anno = seq_dict[key]

        root = ET.Element('annotation')
        folder = ET.SubElement(root, 'sequence')
        folder.text = seq_name
        filename = ET.SubElement(root, 'filename')
        filename.text = '{}.jpg'.format(img_name)
        source = ET.SubElement(root,'source')
        et_img_path = ET.SubElement(source,'img_path')
        img_path = os.path.join(SeqPath,seq_name,img_name+'.jpg')
        et_img_path.text = img_path
        # print(img_path)
        img = cv2.imread(img_path)
        img_h, img_w, img_d = img.shape
        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(img_w)
        height = ET.SubElement(size, 'height')
        height.text = str(img_h)
        depth = ET.SubElement(size, 'depth')
        depth.text = str(img_d)
        segmented = ET.SubElement(root, 'segmented')
        segmented.text = str(0)
        for objitem in img_anno:
            if objitem[6]==0:
                continue

            xmlobject = ET.SubElement(root, 'object')
            objname = ET.SubElement(xmlobject, 'name')
            objname.text = CLASSES[objitem[6]]

            truncation = ET.SubElement(xmlobject, 'truncated')
            truncation.text = str(objitem[6])

            difficult = ET.SubElement(xmlobject, 'difficult')
            if objitem[5] == 0:
                difficult.text = str(1)
            else:
                difficult.text = str(0)

            bndbox = ET.SubElement(xmlobject, 'bndbox')
            xmin = objitem[1]
            xmax = objitem[1] + objitem[3]
            ymin = objitem[2]
            ymax = objitem[2] + objitem[4]
            ET.SubElement(bndbox, 'xmin').text = str(xmin)
            ET.SubElement(bndbox, 'xmax').text = str(xmax)
            ET.SubElement(bndbox, 'ymin').text = str(ymin)
            ET.SubElement(bndbox, 'ymax').text = str(ymax)
        tree = ET.ElementTree(root)
        xml_name = img_name + '.xml'
        xml_path = os.path.join(seq_xml_dir,xml_name)
        print(xml_path)
        tree.write(xml_path)


exit()
score_dict = {}
for anno in AnnoList:
    img_name = anno.split('.')[0]
    annopath = os.path.join(AnnoPath,img_name+'.txt')
    imgpath = os.path.join(ImgPath,img_name+'.jpg')
    with open(annopath) as f:
        root = ET.Element('annotation')
        folder = ET.SubElement(root,'folder')
        folder.text = 'VisDrone2018-DET-val'
        filename = ET.SubElement(root, 'filename')
        filename.text = '{}.jpg'.format(img_name)
        img = cv2.imread(imgpath)
        img_h,img_w,img_d=img.shape
        size = ET.SubElement(root,'size')
        width = ET.SubElement(size,'width')
        width.text = str(img_w)
        height = ET.SubElement(size, 'height')
        height.text = str(img_h)
        depth = ET.SubElement(size, 'depth')
        depth.text = str(img_d)
        segmented = ET.SubElement(root,'segmented')
        segmented.text=str(0)
        for line in f.readlines():
            try:
                objitem = [int(x) for x in line.split(',')]
            except:
                objitem = [int(x) for x in line[:-2].split(',')]
            if objitem[4] in score_dict:
                score_dict[objitem[4]]+=1
            else:
                score_dict[objitem[4]]=1
            if objitem[5]==0:
                continue
            xmlobject = ET.SubElement(root,'object')
            objname = ET.SubElement(xmlobject,'name')
            objname.text=CLASSES[objitem[5]]
            truncation = ET.SubElement(xmlobject,'truncated')
            truncation.text=str(objitem[6])
            difficult = ET.SubElement(xmlobject,'difficult')
            if objitem[4]==0:
                difficult.text = str(1)
            else:
                difficult.text =str(0)
            bndbox = ET.SubElement(xmlobject,'bndbox')
            xmin = objitem[0]
            xmax = objitem[0]+objitem[2]
            ymin = objitem[1]
            ymax = objitem[1]+objitem[3]
            ET.SubElement(bndbox,'xmin').text=str(xmin)
            ET.SubElement(bndbox, 'xmax').text = str(xmax)
            ET.SubElement(bndbox, 'ymin').text = str(ymin)
            ET.SubElement(bndbox, 'ymax').text = str(ymax)
        tree = ET.ElementTree(root)
        xmlpath = os.path.join(xmlDir,img_name+'.xml')
        print(xmlpath)
        tree.write(xmlpath)
print(score_dict)

