import os
import re
import cv2
import numpy as np
from xml.dom import minidom
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt


def get_face_boundary(img):
	dx = [0,1,1,1,0,-1,-1,-1]
	dy = [1,1,0,-1,-1,-1,0,1]
	threshold = 30
	midx = int(expectedHeight/2) 
	midy = midx
	vis = []
	for i in range(0,expectedHeight):
		poo = [0]*expectedHeight
		vis.append(poo)
	vis[midx][midy] = 1
	queue = [] 
	queue.append([midx,midy])
	boundary = []
	mx = [0]*expectedHeight
	mn = [expectedHeight+10]*expectedHeight
	while(len(queue) > 0) :

		x = queue[0][0]
		y = queue[0][1]
		print(x,y)
		queue.pop(0)
		for i in range(8) :
			newx = x + dx[i]
			newy = y + dy[i]
			c = 0
			if(newx >= 0 and newx < expectedHeight and newy >= 0 and newy < expectedHeight) :
				if vis[newx][newy] == 0 and abs(int(img[newx][newy])-int(img[x][y])) <= threshold :
					vis[newx][newy] = 1
					queue.append([newx,newy])
				else :
					c = c + 1
		if c > 0 :
			boundary.append([y,x])
			mn[x] = min(mn[x],y)
			mx[x] = max(mx[x],y)
	print(mn)
	print(mx)
	return boundary


expectedHeight=100
#img = cv2.imread("./cartoonFaces/AamirKhan0001.jpeg",cv2.IMREAD_GRAYSCALE)






i = 0
dic = {}
dic_check = {}
c = 0
ret = []
dir_path = "./fullCartoonImgsAndXMLs"
for filename in os.listdir(dir_path) :
	i = i+1
	print(i)
	filen, file_extension = os.path.splitext(filename)
	#print(file_extension)
	if file_extension == ".xml" :
		continue
	match = re.match(r"([a-z-]+)([0-9]+).([a-z]+)", filename, re.I)
	if match:
	    item = match.groups()
	    # print(item)
	    if item[0] not in dic:
	    	dic[item[0]] = c + 1
	    	c = c + 1
	else :
		assert False
	if item[2] == "xml" :
		continue
	
	img = cv2.imread(dir_path + "/" + filename,cv2.IMREAD_GRAYSCALE)

	obj = minidom.parse(dir_path + "/" + item[0] + item[1] + ".xml")
	obj = obj.getElementsByTagName('zone')
	ulx = max(int(obj[0].attributes['ulx'].value),0)
	uly = max(0,int(obj[0].attributes['uly'].value))
	lrx = max(0,int(obj[0].attributes['lrx'].value))
	lry = max(0,int(obj[0].attributes['lry'].value))
	print(filename)
	# if(filename == "Jay-Z0034.jpeg") :
	# 	print(ulx,uly,lrx,lry,dir_path + "/" + item[0] + item[1] + ".xml")
	# print(type(img))
	cropped_img = img[uly:lry,ulx:lrx]
	#print(cropped_img.shape)

	####################################################
	newAspectRatio=1.0*expectedHeight/len(cropped_img)
	cropped_img=cv2.resize(cropped_img,(0,0),fx=newAspectRatio,fy=newAspectRatio)
	#print(len(cropped_img[0]))
	if(len(cropped_img[0])<expectedHeight):
		dif=(expectedHeight-len(cropped_img[0]))/2
		cropped_img=cv2.copyMakeBorder(cropped_img,0,0,int(dif),expectedHeight-int(dif)-len(cropped_img[0]),cv2.BORDER_CONSTANT,value=255)
	elif(len(cropped_img[0])>expectedHeight):
		cropped_img=cv2.resize(cropped_img,(expectedHeight,expectedHeight))



	####################################################


	dic_check[(cropped_img.shape)] = 1
	# cv2.imshow("cropped",cropped_img)
	# cv2.waitKey(0)
	type(cropped_img)
	#print(cropped_img)
	for i in range(20,60):
		for j in range(20,60):
			print(cropped_img[i][j] ,end = " ")
		print("")
	print("\n===============================\n")
	get_face_boundary(cropped_img)
	plt.imshow(cropped_img, 'gray')
	plt.show()
	#cv2.waitKey()
	#import pdb;pdb.set_trace()
	#get_face_boundary(cropped_img)

	poo = []
	poo.append(cropped_img)
	poo.append(dic[item[0]])
	ret.append(poo)

#print(ret)

'''
dir_path = "./realFaces"
for filename in os.listdir(dir_path) :
	i = i+1
	print(i)
	match = re.match(r"([a-z-]+)([0-9]+).([a-z]+)", filename, re.I)
	if match:
	    item = match.groups()
	    if item[0] not in dic:
	    	assert False 
	else :
		assert False
	
	cropped_img = cv2.imread(dir_path + "/" + filename,cv2.IMREAD_GRAYSCALE)

	####################################################
	newAspectRatio=1.0*expectedHeight/len(cropped_img)
	cropped_img=cv2.resize(cropped_img,(0,0),fx=newAspectRatio,fy=newAspectRatio)
	#print(len(cropped_img[0]))
	if(len(cropped_img[0])<expectedHeight):
		dif=(expectedHeight-len(cropped_img[0]))/2
		cropped_img=cv2.copyMakeBorder(cropped_img,0,0,int(dif),expectedHeight-int(dif)-len(cropped_img[0]),cv2.BORDER_CONSTANT,value=255)
	elif(len(cropped_img[0])>expectedHeight):
		cropped_img=cv2.resize(cropped_img,(expectedHeight,expectedHeight))
	dic_check[(cropped_img.shape)] = 1
	poo = []
	poo.append(cropped_img)
	poo.append(dic[item[0]])
	ret.append(poo)










################################

df = pd.DataFrame(data = ret)
train, test = train_test_split(df, test_size=0.2)
print(len(train))
print(len(test))


with open('train.pkl', 'wb') as f:
	pickle.dump(train, f)
with open('test.pkl', 'wb') as f:
	pickle.dump(test, f)
print(dic_check)

'''