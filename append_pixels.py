import csv
import os
import cv2
import fileinput
import pandas as pd 

dir_path = "/home/saurav/Documents/IIIT-CFW1.0/normalizedFaces"
csv_dir = "landmarks.csv"

all_pixels = []

with open(csv_dir, 'r+', encoding='utf-8') as f:
	reader = csv.reader(f, delimiter=',')
	c = 1
	for data in reader:
		'''
		if c > 1: 
			continue
		c += 1
		'''
	
		filename = data[0].split(',')[0]
		print(filename)
		

		for file in os.listdir(dir_path) :
			if file == filename :
				filen, file_extension = os.path.splitext(filename)

				img = cv2.imread(dir_path + "/" + filename,cv2.IMREAD_GRAYSCALE)
				img= img.tolist()

				#print(img)
				
				img = [item for sublist in img for item in sublist]
				
				all_pixels.append(img)
				
				
#print(all_pixels)
'''
i = -1
def myfunc():
	i = i + 1
	return all_pixels[i]

df = pd.read_csv(csv_dir, sep=',')
df.apply(lambda x: ' '.join(map(string, myfunc())), axis=1)
df.to_csv('new.csv', sep=',', header=True, mode='w', index=False)
'''		
i = 0
with open(csv_dir, 'r+', encoding='utf-8') as f:
	reader = csv.reader(f, delimiter=',')
	with open("train.csv", 'w', newline='') as outf:
		writer = csv.writer(outf, delimiter=',', lineterminator='\n')
		list = all_pixels[i]
		i = i + 1
		for j in reader:
			lis = j[0].split(',')
			lis.extend(list)
			writer.writerow(lis)	

				