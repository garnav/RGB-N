from pycocotools.coco import COCO
import numpy as np
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
from PIL import Image
from PIL import ImageFilter
import argparse
import sys
import time
import pdb
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='input begin and end category')
  parser.add_argument('--begin', dest='begin',
            help='begin type of cat', default=None, type=int)
  parser.add_argument('--end', dest='end',
            help='begin type of cat',
            default=None, type=int)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args
args=parse_args()
print(args.begin)
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
dataDir='cocoapi/images'
dataType='train2014'
annFile='cocoapi/annotations/instances_%s.json'%(dataType)
coco=COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
with open('../filter_tamper/gt/gt_boxes.txt', 'w') as gt_file:
    print('Beginning Data Creation.')
    start = time.time()
    to_write_append = ""
    for cat in cats[args.begin:args.end]:
        print("Creating data for category: " + cat['name'])
        for num in range(2000):
            try:
                catIds = coco.getCatIds(catNms=[cat['name']]);
                imgIds = coco.getImgIds(catIds=catIds );
                img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
                #I = io.imread('http://mscoco.org/images/%d'%(img['id']))
                #I = io.imread(img['coco_url'])
                I=io.imread(os.path.join(dataDir,dataType,'COCO_train2014_{:012d}.jpg'.format(img['id'])))
                #plt.imshow(I); plt.axis('off')
                annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
                anns = coco.loadAnns(annIds)
                #coco.showAnns(anns)
                bbx=anns[0]['bbox']
                mask=np.array(coco.annToMask(anns[0]))
                #print(np.shape(mask))
                #print(np.shape(I))
                #pdb.set_trace()
                I1=I

                #row,col=np.where(mask>0)
                #print(row)
                #print(col)
                #I1=I[row,col,0]
                #print(np.shape(I1))
                #print("I1 shape is: ")
                #print(I1.shape)
                I1[:,:,0]=np.array(I[:,:,0] * mask )
                I1[:,:,1]=np.array(I[:,:,1] * mask )
                I1[:,:,2]=np.array(I[:,:,2] * mask )
                #pdb.set_trace()
                rand=np.random.randint(100,size=1)[0]
                #flag=0
                #=cv2.GaussianBlur(I1,(5,5),0)
                #ag=1
                img1 = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
                #b1 = io.imread('http://mscoco.org/images/%d'%(img1['id']))
                #b1 = io.imread(img1['coco_url'])
                b1=io.imread(os.path.join(dataDir,dataType,'COCO_train2014_{:012d}.jpg'.format(img1['id'])))
                text_img = Image.new('RGBA', (np.shape(b1)[0],np.shape(b1)[1]), (0, 0, 0, 0))
                #print(b1.shape)
                background=Image.fromarray(b1,'RGB')

                # Old Code (slow)
                # foreground=Image.fromarray(I1,'RGB').convert('RGBA')
                # datas=foreground.getdata()
                # #pdb.set_trace()
                # newData = []
                # for item in datas:
                #     if item[0] == 0 and item[1] == 0 and item[2] == 0:
                #         newData.append((0, 0, 0, 0))
                #     else:
                #         newData.append(item)
                # foreground.putdata(newData)

                #print("Using numpy for foreground speedup.")
                # numpy speedup
                foreground_np = np.zeros((I1.shape[0], I1.shape[1], 4), dtype=np.uint8)
                foreground_np[:, :, :3] = I1
                alpha_mask = np.any(foreground_np[:, :, :3] > 0, axis=2)
                foreground_np[:,:,3] = alpha_mask * 255
                foreground = Image.fromarray(foreground_np, 'RGBA')

                height_ratio = background.size[1] / float(foreground.size[1])
                width_ratio = background.size[0] / float(foreground.size[0])

                foreground=foreground.resize((background.size[0],background.size[1]),Image.ANTIALIAS)
                background.paste(foreground,(0,0),mask=foreground.split()[3])

                if rand%3<2:
                    background=background.filter(ImageFilter.GaussianBlur(radius=1.5))

                #print("Pasting complete.")

                x1 = bbx[0]
                y1 = bbx[1]
                x2 = bbx[0]+bbx[2]
                y2 = bbx[1]+bbx[3]

                file_path = '../filter_tamper/Tp_'+str(img['id'])+'_'+str(img1['id'])+'_'+str(x1)+'_'+str(y1)+'_'+str(x2)+'_'+str(y2)+'_'+cat['name']+'.png'

                if not os.path.isfile(file_path):
                    background.save(file_path)
                    to_write = [file_path, str(x1 * width_ratio), str(y1 * height_ratio),
                                           str(x2 * width_ratio), str(y2 * height_ratio), "tamper"]
                    to_write_append += " ".join(to_write) + "\n"

            except Exception as e:
                print(e)
        end = time.time()
        gt_file.write(to_write_append)
        print("Time for this category: {0}".format(end - start))

print('finished')
#I1=np.array([[I[i,j,:] for j in range(len(I[i,:,0])) if mask[i,j]]for i in range(len(I[:,:,0]))])
