import sys
import os

dirname = "/home/saurav/Documents/IIIT-CFW1.0"
test = os.listdir(dirname)

for item in test:
	if item.endswith(".jpg"):
		os.remove(os.path.join(dirname, item))