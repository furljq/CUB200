import cv2
import numpy as np

def read_str(filename):
	tmp = []
	with open(filename, 'r') as file_to_read:
		while True:
			lines = file_to_read.readline()
			if not lines:
				break
			line = lines.split()
			if (len(line)>2):
				tmp.append(line[1]+line[2])
			else:
				tmp.append(line[1])
	return tmp

def read_ints(filename, n, m):
	tmp_loc = np.zeros((n, m, 2))
	tmp_appear = np.zeros((n, m))
	with open(filename, 'r') as file_to_read:
		while True:
			lines = file_to_read.readline()
			if not lines:
				break
			line = lines.split()
			tmp_loc[int(line[0])-1, int(line[1])-1, 0] = float(line[2])
			tmp_loc[int(line[0])-1, int(line[1])-1, 1] = float(line[3])
			tmp_appear[int(line[0])-1, int(line[1])-1] = int(line[4])
	return tmp_loc, tmp_appear

class cropper:
	def __init__(self):
		self.image_dirs = read_str('./CUB_200_2011/images.txt')
		self.num_images = len(self.image_dirs)

		self.parts = read_str('./CUB_200_2011/parts/parts.txt')
		self.num_parts = len(self.parts)
		self.part_locs,	self.part_appear = read_ints('./CUB_200_2011/parts/part_locs.txt', self.num_images, self.num_parts)

		self.part_size = [(0.3,0.3), (0.15, 0.15), (0.2, 0.2), (0.15, 0.15), (0.2, 0.2), (0.15, 0.15), (0.1, 0.1),
					   (0.2, 0.2), (0.3, 0.3), (0.2, 0.2), (0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.25, 0.25), (0.2, 0.2)]
		self.joints = [(1,3), (1,14), (1,10), (2,6), (3,8), (3,12), (3,4),
					   (4, 15), (4,9), (4,13), (5,6), (6,7), (6,11), (6, 15), (10, 15)]

	def get_body_len(self, parts):
		res = 0
		for a in parts:
			for b in parts:
				l = np.linalg.norm(a - b)
			res = max(res, l)
		return res

	def shift(self, l, r, img_size):
		if l < 0:
			l, r = 0, r - l
		if r >= img_size:
		    l, r = l - r + img_size - 1, img_size - 1
		if l < 0:
			l = 0
		return l, r

	def crop(self, image_id, part_id):
		part_appear = self.part_appear[image_id, part_id]
		if not part_appear:
			return np.zeros((224, 224, 3)), False
		img = cv2.imread('./CUB_200_2011/images/'+self.image_dirs[image_id])
		x, y = self.part_locs[image_id, part_id]
		body_len = self.get_body_len(self.part_locs[image_id])
		w, h = self.part_size[part_id]
		w, h = int(w * body_len), int(h * body_len)
		xl, xr, yl, yr = int(y)-h//2, int(y)+h//2, int(x)-w//2, int(x)+w//2
		img_size = img.shape[0:2]
		xl, xr = self.shift(xl, xr, img_size[0])
		yl, yr = self.shift(yl, yr, img_size[1])
		subimg = img[xl:xr, yl:yr, :]
		return cv2.resize(subimg, (224, 224)), True

	def draw_skeleton(self, image_id):
		img = cv2.imread('./CUB_200_2011/images/'+self.image_dirs[image_id])
		for j in range(self.num_parts):
			x,y = self.part_locs[image_id][j]
			part_appear = self.part_appear[image_id][j]
			if part_appear:
				cv2.circle(img, (int(x),int(y)), radius=5, color=(0,0, 255), thickness=-1)
		#for (p1, p2) in self.joints:
		for p1 in range(self.num_parts):
			for p2 in range(self.num_parts):
				if self.part_appear[image_id][p1] and self.part_appear[image_id][p2]:
					cv2.line(img, (int(self.part_locs[image_id][p1][0]), int(self.part_locs[image_id][p1][1])),
							 (int(self.part_locs[image_id][p2][0]), int(self.part_locs[image_id][p2][1])), (0, 255, 0))
		print(image_id)
		cv2.imshow("Image", img)
		cv2.waitKey(1000)
