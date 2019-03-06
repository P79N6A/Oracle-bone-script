import os
import random

oraclePath = "C:/Users/24400/Desktop/oracle"

jinPath = "C:/Users/24400/Desktop/jin"

numList = []

oracles = os.listdir(oraclePath)
for oracle in oracles:
	tempPath = oraclePath+"/"+oracle
	tempJin = jinPath+"/"+oracle
	oracleNum = os.listdir(tempPath)
	oracleNum = len(oracleNum)
	jinNum = os.listdir(tempJin)
	jinNum = len(jinNum)
	j=0
	while j!= oracleNum:
		numList.append(jinNum)
		j+=1

print(numList)


