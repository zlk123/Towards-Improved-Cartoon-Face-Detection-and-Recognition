import os 
import cv2
import csv
dir_path = "./reducedSet"

ret = []
all_pixels = []


for filename in os.listdir(dir_path) :
	img = cv2.imread(dir_path + "/" + filename,cv2.IMREAD_GRAYSCALE)
	img = img.tolist()
	img = [item for sublist in img for item in sublist]
	'''
	poo = ""
	poo += filename 
	poo += "," + " ".join(map(str,img))
	x = [] 
	x.append(poo)
	ret.append(x)
	'''

	ret.append([filename])
	all_pixels.append(img)
	#print(len(ret))
	#print(len(all_pixels))


with open('test.csv','w', newline='') as f:
    writer = csv.writer(f, delimiter =',')
    labels = "Filename,Image"
    writer.writerow(labels.split(','))
    
    for i in range(len(ret)):
	    lis = ret[i]
	    mylist = ' '.join(str(e) for e in all_pixels[i])

	    lis.append(mylist)
	    writer.writerow(lis)
    