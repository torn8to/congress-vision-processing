import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL import GifImagePlugin
import io


def gifToImagePath(gifPath):
	gif = Image.open(gifPath)
	gif.save('image1.png')


def gifToImageBytes(gifBytes):
	gif = Image.open(io.BytesIO(gifBytes))
	gif.save("temp.png") #TODO change to a tempfile



def alignImages(im1, template_image):
	im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2Gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
	orb = cv2.ORB().create(nfeatures=500),
	keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
	keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(descriptors1, descriptors2, None)
	matches.sort(key=lambda x: x.distance, reverse=False)
	numGoodMatches = int(len(matches) * .15)
	matches = matches[:numGoodMatches]
	#imMatches = cv2.drawMatches(im1, keypoints1, template_image, keypoints2, matches, None)
	#cv2.imwrite("matches.jpg", imMatches)
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)
	for i, match in enumerate(matches):
		points1[i, :] = keypoints1[match.queryIdx].pt
		points2[i, :] = keypoints2[match.trainIdx].pt
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
	height, width, channels = template_image.shape
	im1Reg = cv2.warpPerspective(im1, h, (width, height))
	return im1Reg, h


def isProperDocument(im1: np.ndarray, template_image: np.ndarray, good_match_limit:int = 23) -> pd.DateFrame():
	"""
	checks that the form is im1 is simple to template_image
	:param im1:
	:param template_image:
	
	:return:
	"""
	im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2Gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
	orb = cv2.ORB().create(nfeatures=500)
	keypoints1, descriptors1 = orb.detectAndCompute(template_image, None)
	keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
	matcher = cv2.DescriptorMatcher().create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
	good_matches = [m for m, n in matches if m.distance < .75 * n.distance]
	return len(good_matches) > good_match_limit


def getHorizontalImageContours(img:np.ndarray,dilation_iterations=1,line_itrerations=3)-> (np.ndarray, list):
	horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[0]//25, 1))
	erosion_dilation_kernel = np.ones((5, 5), np.uint8) * 255
	dilated = cv2.dilate(img, erosion_dilation_kernel, iterations=dilation_iterations)
	detect_horizontal = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, horizontal_kernel, iterations=line_itrerations)
	eroded_detect_horizontal = cv2.erode(detect_horizontal,erosion_dilation_kernel, iterations=1)
	detect_horizontal = cv2.Canny(eroded_detect_horizontal, 50, 150)
	lines = cv2.HoughLinesP(detect_horizontal, 1, np.pi/4, threshold=200, minLineLength=img.shape[1]//12)
	return detect_horizontal, lines


def getVerticalImageContours(img:np.ndarray, dilation_iterations=1, vertical_iterations=3) -> (np.ndarray,list):
	vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img.shape[1]//12))
	erosion_dilation_kernel = np.ones((5, 5), np.uint8)
	dilated = cv2.dilate(img, erosion_dilation_kernel, iterations=3)
	detect_vertical = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, vertical_kernel, iterations=vertical_iterations)
	eroded_detect_horizontal = cv2.erode(detect_vertical, erosion_dilation_kernel, iterations=1)
	detect_horizontal = cv2.Canny(eroded_detect_horizontal, 100, 200)
	lines = cv2.HoughLinesP(detect_vertical, 1, np.pi/4, threshold=200, minLineLength=200)
	return detect_vertical, lines


def getBinsAverage(bins, orientation=0):
	average_list = []
	for bin in bins:
		line_cache = []
		for line in bin:
			line_cache.append(line[orientation])
		average_list.append(int(sum(line_cache) / len(line_cache)))
	return average_list


def binLines(lines: list[list[list]], width=37, orientation=0):
	dynamic_bins = [[lines[0][0]]]
	lines.pop(0)
	for line in lines:
		averages = getBinsAverage(dynamic_bins, orientation=orientation)
		appended_line = False
		for x in range(len(averages)):
			if averages[x] - width < abs(line[0][orientation]) and averages[x] + width > abs(line[0][orientation]):
				dynamic_bins[x].append(line[0])
				appended_line = True
		if appended_line == False:
			dynamic_bins.append(line)
	return dynamic_bins, getBinsAverage(dynamic_bins, orientation=orientation)



