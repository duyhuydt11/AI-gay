import csv
import numpy as np

number = 0
x  = []
y = []
with open('ex1.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file) # chi in duoc gia tri cua tung do khi chay for

    for row in csv_reader:
        xline = row[0]
        yline = row[1]
        x.append(xline)
        y.append(yline)
   
x[0] = y[0] = 0 
print(x)
print(y) 
for i in x:
    tran = int(x)
    x[i] = int(tran)