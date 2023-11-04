import cv2
import numpy as np
import pandas as pd
import pytesseract
import matplotlib.pyplot as plt
from imageProcessing import alignImages, isProperDocument
from imageProcessing import getHorizontalImageContours, getVerticalImageContours, binLines
from tqdm import tqdm
import pickle
from sklearn.neighbors import KNeighborsClassifier
from congressionalDocPull import senatePullImagesPNGConvert
from loguru import logger



money_definition = ["number",
										"full stock name",
										"purchase",
										"sale",
										"exchange",
										"TransactionDate",
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
										#"money_l"
										]


#image to makes sure the document is accurate
template_image = cv2.imread("image1.png")

'''
assumptions
assumed vertical aligment
and readable text
'''
def parse_document(img:np.ndarray) -> pd.DataFrame:
	'''
	:param img:  the document image
	:return: a data frame reconstruction of the dataframe
	'''
	img = 255 - img
	H = img.shape[0]
	hor_img, hor_lines = getHorizontalImageContours(img)
	ver_img, ver_lines = getVerticalImageContours(img)
	horizontalBins, horizontalBinAverages = binLines(hor_lines.tolist(), width=20, orientation=1)
	verticalBins, verticalBinAverages = binLines(ver_lines.tolist(), width=20, orientation=0)
	verticalBinAverages.sort()
	horizontalBinAverages.sort()
	cleanedHorizontalBinAverages = [x for x in horizontalBinAverages if x > H // 2 and x < H - H // 20]
	horizontalBinAverages = cleanedHorizontalBinAverages
 	logger.info((horizontalBinAverages,len(horizontalBinAverages)))
	logger.info((verticalBinAverages,len(verticalBinAverages)))
	
	assert len(horizontalBinAverages) == 12
	assert len(verticalBinAverages) == 18 # asserts that we have the desired table dimensions of the bottom part of the form -> only really needed because of the
	used_column_names:list = []
	counter:int = 0
	data_big:dict = {}
	
	
	neighborsDetector = pickle.load(open('data and objects/neigbors_model.pkl',"rb"))
	for x in tqdm(range(1, len(verticalBinAverages)), desc="rows progress"):
		column:list = []
		col_val:int = 0
		column_name = money_definition[x - 1]
		used_column_names.append(column_name)
		for y in range(1, len(horizontalBinAverages)):
			coords_x, coords_y = (horizontalBinAverages[y - 1], horizontalBinAverages[y]), (verticalBinAverages[x - 1], verticalBinAverages[x])
			ocr_raw = img[coords_x[0]:coords_x[1], coords_y[0]:coords_y[1]]
			ocr_processed = cv2.fastNlMeansDenoising(ocr_raw, None, h=3, templateWindowSize=7, searchWindowSize=21)#get rid of noise on the prolly should just use a gaussian blur
			data = pytesseract.image_to_string(ocr_processed)
			for s in "\n|~:":
				data = data.replace(s, "")
			if len(data) == 0:
				#TODO alter the knn mdoel to have less inputs
				resized = cv2.resize(ocr_raw, (28, 28))
				#cannied = cv2.Canny(ocr_raw,50, 150)#doing this on trianing did not imporve performcnace on the dataset quite the opposite even after retraining big L
				model_input = np.array([resized.flatten()])
				prediction = neighborsDetector.predict(model_input)
				data = prediction
			# plt.imshow(ocr_processed)
			# plt.show()
			column.append(data)
			counter += 1
		data_big[column_name] = column
	stock_tickers = []
	stock_short_names = []
	df = pd.DataFrame(data_big, columns=used_column_names)
	return df

	
def getTicker(row):
	name:str = row["full stock name"]
	words = name.split(" ")
	return words[len(words)-1].strip(" ()")

def getAmount(row:pd.DataFrame):
	money_columns = ["money_A","money_B","money_C",
										"money_D",
										"money_E",
										"money_F",
										"money_G",
										"money_H",
										"money_I",
										"money_J",
										"money_K",
										]
	
	
	full_name = row["full stock name"]
	date = row["TransactionDate"]
	
	switcher = {
		"money_A":"$1,001-$15,000",
		"money_B":"$15,001-$50,000",
		"money_C":"$50,001-$100,000",
		"money_D":"$100,001-$250,000",
		"money_E":"$250,001-$500,000",
		"money_F":"$500,001-$1,000,000",
		"money_G":"$1,000,001-$5,000,000",
		"money_H":"$1,000,001-$5,000,000",
		"money_I":"$5,000,001-$25,000,000",
		"money_J":"$25,000,001-$50,000,00",
		"money_K":">$500,000,000",
		"money_l":">5000,000,000"
	}
	
	for x in money_columns:
		if str(row[x]) == '[1]':
			return switcher.get(x, "$1,001-$15,000")
	logger.error(f"money not found for {full_name} on {date}")
	return "MONEYERROR"
			
	
def getTransactionType(row):
	full_name = row["full stock name"]
	date = row["TransactionDate"]

	if str(row["purchase"]) == '[1]':
		return 'P'
	elif str(row["sale"]) == '[1]':
		return 'S'
	elif str(row["exchange"]) =='[1]':
			return 'E'
	else:
		logger.error(f"purchase not found on  {full_name} on {date}")
		return 'TRANSACTIONERROR'


def postProcessing(df:pd.DataFrame()) -> pd.DataFrame():
		column_names = df.keys()
		publicly_traded = df["full stock name"].str.lower().str.contains('stock').fillna(False)
		publicly_traded_rows = df[publicly_traded]
		ticker, transactionDate,amount, transaction = [], [], [], []
		for x in range(len(publicly_traded_rows)):
			rowTicker = getTicker(publicly_traded_rows.iloc[x])
			rowTransaction = getTransactionType(publicly_traded_rows.iloc[x])
			rowAmount = getAmount(publicly_traded_rows.iloc[x])
			rowDate = publicly_traded_rows.iloc[x]['TransactionDate']
			ticker.append(rowTicker)
			transactionDate.append(rowTransaction)
			amount.append(rowAmount)
			transaction.append(rowDate)
			
		dictionary = {
			"Ticker":ticker,
			"TransactionDate":transactionDate,
			"Amount":amount,
			"Transaction":transaction
		}
		return pd.DataFrame(dictionary)
	
		
def parseDocuments(imgs:[np.ndarray]):
	template_image = cv2.imread("image1.png")
	'''
	useful = [x for x in imgs if isProperDocument(x, template_image)] # drops non documents prolly needs print statement to warnin case multiple documents are dopped TODO
	aligned_images = []
	for x in useful:
		aligned, h = alignImages(x, template_image)
		aligned_images.append(aligned_images)
	'''
	#obscured but wokrs for documents that are roatated slightly
	aligned_images = imgs
	dfs = pd.DataFrame()
	for x in aligned_images:
			dfs = pd.concat([dfs, postProcessing(parse_document(x))], ignore_index=True)
	return dfs

'''
def parseFromSenatePull():
	imgs = senatePullImagesPNGConvert()
	return parseDocuments(imgs)
'''

if __name__ == "__main__":
	#df = pd.read_csv("output.csv",usecols=money_definition)
	template_image = cv2.imread("image1.png",0)
	out:pd.DataFrame = postProcessing(parse_document(template_image))
	print(out)
	out.to_csv("output.csv")

