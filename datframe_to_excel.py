import numpy as np 
import pandas as pd 


if __name__ == '__main__':

	dataset = "indian_pines_corrected"
	numComponent = 32
	windowSize = 9
	testRatio = 0.95
	dost_used = True
	wls_alpha = 1.2
	wls_lambda = 0.8
	
	
	filename = "conv_{}_{}_{}_{}_{}_{}_{}_1".format(dataset,numComponent,windowSize,testRatio,dost_used,wls_alpha,wls_lambda)

	df = pd.read_pickle("../excelsheets/"+filename+".pkl")

	print(df)

	writer = pd.ExcelWriter("../excelsheets/"+filename+".xlsx")
	df.to_excel(writer,'sheet1')

