import cv2
import numpy as np
import pandas as pd
import pytesseract
import requests
from imageProcessing import getHorizontalImageContours, getVerticalImageContours, binLines
import pickle
from tqdm import tqdm
from pdf2image import convert_from_bytes



neighbors_image_detector = pickle.load(open("data and objects/neigbors_model.pkl", "rb"))
money_definition = ["sp/dc/jt",
										"full stock name",
										"purchase",
										"sell",
										"exchange",
										"cap gains exchange 200",
										"Partial transaction",
										"Transaction Date",
										"Notification Date",
										"money_A",
										"money_B",
										"money_C",
										"money_D",
										"money_E",
										"money_F",
										"money_G",
										"money_H",
										"money_I",
										"money_J",
										"money_K",
										"money_l" ]

def bin_lines_horizontal(lines):
	return binLines(lines, width=80, orientation=1)


def bin_lines_vertical(lines):
	return binLines(lines, width=40, orientation=0)


def get_ticker_and_short_name(company_name):
	yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
	user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
	params = {"q": company_name, "quotes_count": 1, "country": "United States"}
	res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
	data = res.json()
	company_code = data['quotes'][0]['symbol']
	short_name = data['quotes'][0]['shortname']
	return company_code, short_name


def parse_page(img_in, report_date=""):
	img_read = img_in
	img = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
	# img = cv2.fastNlMeansDenoising(img,None,10,10,7,21)
	read_rotaated = cv2.rotate(img_read, cv2.ROTATE_90_COUNTERCLOCKWISE)
	img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
	base_img = cv2.GaussianBlur(img, (5, 5), 0)
	T, thresh = cv2.threshold(base_img, 200, 255, cv2.THRESH_BINARY_INV)
	base_img = thresh
	processes_image_vertical, vertical_features = getVerticalImageContours(base_img)
	processes_image_horizontal, horizontal_features = getHorizontalImageContours(base_img)
	vertical_bins, vertical_averages = bin_lines_vertical(vertical_features.tolist())
	horizontal_bins, horizontal_averages = bin_lines_horizontal(horizontal_features.tolist())
	vertical_averages.sort()
	horizontal_averages.sort()
	
	pruned_horizontal_averages = [b for b in horizontal_averages if b > 600]
	H, W, D = read_rotaated.shape
	resized_img = cv2.resize(read_rotaated, (W // 4, H // 4))
	data_big = {}
	used_column_names = []
	counter = 0
	print(vertical_averages)
	print(horizontal_averages)
	for x in tqdm(range(1, len(vertical_averages)), desc="rows progress"):
		column = []
		col_val = 0
		column_name = money_definition[x - 1]
		used_column_names.append(column_name)
		for y in range(1, len(pruned_horizontal_averages)):
			coords_x, coords_y = (pruned_horizontal_averages[y - 1], pruned_horizontal_averages[y]), (vertical_averages[x - 1], vertical_averages[x])
			ocr_raw = img[coords_x[0]:coords_x[1], coords_y[0]:coords_y[1]]
			ocr_processed = cv2.fastNlMeansDenoising(ocr_raw, None, h=3, templateWindowSize=7, searchWindowSize=21)
			data = pytesseract.image_to_string(ocr_processed)
			for s in "\n|~:":
				data = data.replace(s, "")
			if len(data) == 0:
				resized = cv2.resize(ocr_processed, (28, 28))
				model_input = np.array([resized.flatten()])
				prediction = neighbors_image_detector.predict(model_input)
				data = prediction
			# plt.imshow(ocr_processed)
			# plt.show()
			column.append(data)
			counter += 1
		
		data_big[column_name] = column
	stock_tickers = []
	stock_short_names = []
	df = pd.DataFrame(data_big, columns=used_column_names)
	# money_columns = df[[x for x in money_definition if "money" in x:]]
	for c in df["full stock name"]:
		try:
			result, short_name = get_ticker_and_short_name(c[:len(c) // 3])
		except:
			result, short_name = np.NAN, np.NAN
		stock_tickers.append(result)
		stock_short_names.append(short_name)
	df2 = df.copy()
	df["Ticker"] = stock_tickers
	df["ShortName"] = stock_short_names
	if report_date != None:
		df["ReportDate"] = report_date
	return df


def parseFromHouseDataFrameRow(row):
	#row is a dataframee row that has been iterated through
	base_url:str = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs"
	request_url:str = base_url + f"/{row.Year}/{row.DocID}.pdf"
	req = requests.get(request_url)
	imgs = convert_from_bytes(req.content, dpi=400, fmt="png", poppler_path="poppler/poppler-23.08.0/Library/bin")
	
	
	
	