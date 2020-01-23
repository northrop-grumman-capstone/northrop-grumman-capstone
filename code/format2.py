#!/usr/bin/env python3
import os
import argparse
import csv
import random
import cv2
import time
import pickle
import utils
import numpy as np
import traceback
from joblib import Parallel, delayed

data1 = dict() # url -> timestamp -> [[cid, oid, xmin, xmax, ymin, ymax], ...]
data2 = dict() # url+cid+oid -> start_time
out_fps = None
out_width = None
out_height = None

def parse_csv(anno_file):
	### Parse the CSV file
	with open(anno_file,newline="") as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			uid = "{}+{}+{}.mp4".format(row[0], row[2], row[4])
			timestamp = int(row[1])
			if(uid not in data2): # figure out timestamp at which video starts
				data2[uid] = timestamp
			if(row[5]=="present"): # object is present in frame
				if(row[0] not in data1):
					data1[row[0]] = dict()
				if(timestamp not in data1[row[0]]):
					data1[row[0]][timestamp] = []
				data1[row[0]][timestamp].append([int(row[2]), int(row[4]), float(row[6]), float(row[7]), float(row[8]), float(row[9])])

def process_video(infolder, fname, out_anno_file, out_vid_folder):
	try:
		os.makedirs(out_vid_folder)
		url = fname.split("+")[0]
		start_time = data2[fname]
		video = cv2.VideoCapture(infolder+"/"+fname)
		if(not video.isOpened()):
			print("Failed to open video: {}".format(fname))
			return
		frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
		frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
		fps = round(video.get(cv2.CAP_PROP_FPS))
		between_writes = int(fps/out_fps)
		if(between_writes==0): between_writes = 1 # Some videos have fps as low as 7
		frame_num = 0
		annotations = []
		while(True): # Iterate through video, track objects, extract/resize frames, write out numpy arrays, and create annotations
			ok, frame = video.read()
			if(not ok):
				break
			if(frame_num%fps==0): # Get accurate data from annotations each second
				t = start_time+(1000*frame_num/fps)
				if(url in data1 and t in data1[url]):
					bboxes = [(data[0], utils.bbox_to_rect(data[2:], frame_width, frame_height)) for data in data1[url][t]]
				else: bboxes = []
				trackers = [cv2.TrackerMOSSE_create() for box in bboxes]
				for i in range(0, len(trackers)):
					trackers[i].init(frame, bboxes[i][1])
			else: # Propagate bounding boxes
				for i in range(0,len(trackers)):
					ok, bbox = trackers[i].update(frame)
					if(ok):
						bboxes[i] = (bboxes[i][0], bbox)
					else:
						bboxes[i] = (bboxes[i][0], None)
			if(frame_num%between_writes==0):
				img = cv2.resize(frame, (out_width, out_height))
				np.save(out_vid_folder+"/"+str(int(frame_num/between_writes)), img)
				annotations.append([(bbox[0], utils.rect_to_bbox(bbox[1], frame_width, frame_height)) for bbox in bboxes if bbox[1]!=None])
			frame_num += 1
		with open(out_anno_file, 'wb') as f:
			pickle.dump(annotations, f)
	except Exception:
		print(traceback.format_exc())
		
def main():
	global out_fps, out_width, out_height
	parser = argparse.ArgumentParser(description="Format Youtube-BB data into something useful.")
	parser.add_argument("anno_file", help="The csv file containing Youtube-BB annotations")
	parser.add_argument("dataset_folder", help="folder of dataset to process")
	parser.add_argument("out_folder", help="folder to output into")
	parser.add_argument("--fps", type=int, default=5, help="fps at which to extract frames (default 5)")
	parser.add_argument("--width", type=int, default=-1, help="width of extracted frames (default to height (or 96 if not given))")
	parser.add_argument("--height", type=int, default=-1, help="height of extracted frames (default to width)")
	parser.add_argument("--split", default="70,15,15", help="proportion of dataset to be used for training, validation, and testing (default 70,15,15)")
	args = parser.parse_args()
	if(args.width==-1 and args.height==-1): args.width = 96
	if(args.width==-1): args.width = args.height
	if(args.height==-1): args.height = args.width
	out_fps = args.fps
	out_width = args.width
	out_height = args.height
	split = [int(s) for s in args.split.split(',')]
	if(len(split)!=3 or sum(split)!=100):
		print("Invalid split")
		return
	try:
		os.makedirs(args.out_folder+"/train/videos")
		os.makedirs(args.out_folder+"/train/annots")
		os.makedirs(args.out_folder+"/valid/videos")
		os.makedirs(args.out_folder+"/valid/annots")
		os.makedirs(args.out_folder+"/test/videos")
		os.makedirs(args.out_folder+"/test/annots")
	except:
		print("Output folders already exist")
		return
	parse_csv(args.anno_file)
	print("Parsed CSV")
	### Iterate over videos
	train_videos = []
	valid_videos = []
	test_videos = []
	for item in os.walk(args.dataset_folder):
		random.shuffle(item[2])
		for i in range(0, len(item[2])):
			if(item[2][i].endswith(".mp4")):
				if(i<len(item[2])*split[0]/100):
					train_videos.append((item[0], item[2][i]))
				elif(i<len(item[2])*(split[0]+split[1])/100):
					valid_videos.append((item[0], item[2][i]))
				else:
					test_videos.append((item[0], item[2][i]))
	anno_file = args.out_folder+"/{}/annots/{}.pickle"
	vid_folder = args.out_folder+"/{}/videos/{}"
	print("Starting Training Videos...")
	Parallel(n_jobs=-1, backend="multiprocessing")(delayed(process_video)(
		item[1][0], item[1][1], anno_file.format("train", item[0]), vid_folder.format("train", item[0]))
		for item in enumerate(train_videos))
	print("Finished Training Videos")
	print("Starting Validation Videos...")
	Parallel(n_jobs=-1, backend="multiprocessing")(delayed(process_video)(
		item[1][0], item[1][1], anno_file.format("valid", item[0]), vid_folder.format("valid", item[0]))
		for item in enumerate(valid_videos))
	print("Finished Validation Videos")
	print("Starting Testing Videos...")
	Parallel(n_jobs=-1, backend="multiprocessing")(delayed(process_video)(
		item[1][0], item[1][1], anno_file.format("test", item[0]), vid_folder.format("test", item[0]))
		for item in enumerate(test_videos))
	print("Finished Testing Videos")


if __name__ == "__main__":
	main()
