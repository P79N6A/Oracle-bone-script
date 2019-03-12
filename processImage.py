import os,shutil,stat

path="C:/Users/24400/Desktop/result/"

parents = os.listdir(path)

jinList = []
oracleList = []

i=0
j=0
for parent in parents:
	tempPath = path+parent

	if os.path.exists(tempPath+"/甲骨文"):
		
		destPath = "C:/Users/24400/Desktop/oracle-all"
		destPath = destPath+"/"+parent
		os.mkdir(destPath)

		filePath = tempPath+"/甲骨文/png"
		fileList = os.listdir(filePath)
		for file in fileList:
			tempFile = filePath+"/"+file
			reachPath = destPath+"/"+file
			shutil.copyfile(tempFile,reachPath)
			j+=1
			print(j)
		'''
		destPath = "C:/Users/24400/Desktop/jin"
		destPath = destPath+"/"+parent
		os.mkdir(destPath)

		filePath = tempPath+"/金文/png"
		fileList = os.listdir(filePath)
		for file in fileList:
			tempFile = filePath+"/"+file
			reachPath = destPath+"/"+file
			shutil.copyfile(tempFile,reachPath)

		'''
	i+=1
	print(i)