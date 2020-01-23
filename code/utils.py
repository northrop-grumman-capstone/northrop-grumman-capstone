import os
import cv2
import numpy as np
import pickle

classes = ["Person", "Bird", "Bicycle", "Boat", "Bus", "Bear", "Cow", "Cat", "Giraffe",
           "Potted Plant", "Horse", "Motorcycle", "Knife", "Airplane", "Skateboard",
           "Train", "Truck", "Zebra", "Toilet", "Dog", "Elephant", "Umbrella", "None", "Car"]

def bbox_to_rect(bbox, width, height): #bbox is [xmin, xmax, ymin, ymax] as floats between 0 and 1
	return (int(bbox[0]*width), int(bbox[2]*height), int((bbox[1]-bbox[0])*width), int((bbox[3]-bbox[2])*height))

def rect_to_bbox(rect, width, height): #rect is (xmin, ymin, width, height) as ints in img coords
	return [rect[0]/width, (rect[0]+rect[2])/width, rect[1]/height, (rect[1]+rect[3])/height]

def show(img, title="image", time=0, destroy=True):
	cv2.imshow(title, img)
	k = cv2.waitKey(time)
	if(destroy): cv2.destroyAllWindows()
	return k

def draw_bbox(img, xy1, xy2, color=(255,0,0), label="", t_color=(0,0,0), thickness=4, t_scale=1, t_thickness=2):
	color = (color[2], color[1], color[0])
	t_color = (t_color[2], t_color[1], t_color[0])
	bbox_img = img.copy()
	cv2.rectangle(bbox_img, xy1, xy2, color, thickness=thickness)
	if(label!=""):
		box, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, t_scale, t_thickness)
		hb = int(baseline/2)
		box = (box[0]+baseline, box[1]+baseline)
		cv2.rectangle(bbox_img, (xy1[0],xy1[1]), (xy1[0]+box[0],xy1[1]+box[1]), color, -1)
		cv2.putText(bbox_img, label, (xy1[0]+hb,xy1[1]+box[1]-hb), cv2.FONT_HERSHEY_SIMPLEX, t_scale, t_color, t_thickness) #TODO ????
	return bbox_img

def load_anno(filename):
	with open(filename, 'rb') as f:
		return pickle.load(f)

def play_video(filename, frame_time, title="video", annotations=None, class_labels=False):
	video = cv2.VideoCapture(filename)
	frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
	frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
	if(not video.isOpened()):
		print("Failed to open video")
	while(True):
		ok, frame = video.read()
		frame_num = video.get(cv2.CAP_PROP_POS_FRAMES)-1
		if(not ok): break
		if(annotations != None and len(annotations)>frame_num): #TODO add annotation to frame
			#TODO different color for each bbox
			for anno in annotations[int(frame_num)]:
				xy1 = (int(anno[1][0]*frame_width), int(anno[1][2]*frame_height))
				xy2 = (int(anno[1][1]*frame_width), int(anno[1][3]*frame_height))
				if(class_labels):
					frame = draw_bbox(frame, xy1, xy2, label=classes[anno[0]])
				else:
					frame = draw_bbox(frame, xy1, xy2)
		cv2.imshow(title, frame)
		k = cv2.waitKey(frame_time)
		if(k == 97): # go back one frame when 'a' is pressed
			video.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1 if frame_num>0 else 0)
		if(k == 120): # exit video when 'x' is pressed (forward otherwise)
			break
	try: cv2.destroyWindow(title)
	except: pass

def play_formatted(foldername, frame_time, title="video", annotations=None, class_labels=False, out_size=None):
	frame_num = 0
	while(True):
		try:
			frame = np.load("{}/{}.npy".format(foldername, frame_num))
			if(out_size!=None):
				frame = cv2.resize(frame, out_size)
			frame_height = frame.shape[0]
			frame_width = frame.shape[1]
			if(annotations != None and len(annotations)>frame_num): #TODO add annotation to frame
				#TODO different color for each bbox
				for anno in annotations[int(frame_num)]:
					xy1 = (int(anno[1][0]*frame_width), int(anno[1][2]*frame_height))
					xy2 = (int(anno[1][1]*frame_width), int(anno[1][3]*frame_height))
					if(class_labels):
						frame = draw_bbox(frame, xy1, xy2, label=classes[anno[0]])
					else:
						frame = draw_bbox(frame, xy1, xy2)
			cv2.imshow(title, frame)
			k = cv2.waitKey(frame_time)
			if(k == 97): # go back one frame when 'a' is pressed
				frame_num = frame_num-1 if frame_num>0 else 0
			if(k == 120): # exit video when 'x' is pressed (forward otherwise)
				break
			else:
				frame_num += 1
		except: break
	try: cv2.destroyWindow(title)
	except: pass
