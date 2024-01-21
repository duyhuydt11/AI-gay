import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
############################################################### cau 1 ######################################################################################

#xac dinh so hang hien thi toi da khi in khong bi cat gon
pd.options.display.max_rows = 100 

Table = pd.read_csv('ex1.csv')

# truy xuat du lieu theo cot x va y trong file excel, chuyen du lieu ve dang ma tran
X = Table['x'].to_numpy()
Y = Table['y'].to_numpy()

#doi ma tran lai resharp(hang,cot)
X_matrix = X.reshape(100,1)    
Y_matrix = Y.reshape(100,1) 


print(" Ma tran [100x1] chua du lieu cua cot X: \n")
print(X_matrix)

print('\n')

print(" Ma tran [100x1] chua du lieu cua cot Y: \n")
print(Y_matrix)

############################################################### cau 2 ######################################################################################

# Khởi tạo mảng cơ bản
Thelta_zeros = []  #thelta0
Thelta_Ones = []  #thelta1
J_arr = []

# khai báo các giá trị
Lr = 0.0004 #learning rate nam trong khoang 0 - 1.0
m = 100 #Number of Training Example
num_loop = 10
pos = 1

# chuyen doi ma tran x[m,1] => x[m,2]
col = np.ones((100,1))
X_final = np.hstack((X_matrix,col))

Thelta_update = np.random.randn(2,1)

#khoi tao font
font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}


#ham cost funtion
def Cost_function(h):
    J = (((h-Y_matrix).T) @ (h-Y_matrix))/(2*m) 
    J_arr.append(J[0,0])
    return J;

# khoi tao cac cong thuc cua Gradient_descent
def Gradient_descent(Thelta_update, h):

    Thelta_J = Thelta_update - Lr*((X_final.T) @ (h-Y_matrix))/m
    Thelta_update = Thelta_J

    # luu su thelta lai
    Thelta_zeros.append(Thelta_update[0,0])
    Thelta_Ones.append(Thelta_update[1,0])

    return Thelta_update;

# ve ham cost
def Drawing_cost(Thelta, j):
    plt.subplot(1, 2, 1)
    plt.plot(Thelta, j, linewidth = '2.5')
    plt.title("J(Thelta1) chart", fontdict = font1)
    plt.ylabel("J (Thelta1)", fontdict = font2)
    plt.xlabel("Thelta1", fontdict = font2)
    return;

# ve ham linear va cac diem du lieu
def Drawing_linear(h):
    plt.subplot(1, 2, 2)
    plt.plot(X_matrix, Y_matrix, 'o', ms = 5)    
    plt.plot(X_matrix, h, c = 'red', linewidth = '4')
    plt.title("h (Thelta1) chart", fontdict = font1)
    plt.ylabel("h (Thelta1)", fontdict = font2)
    plt.xlabel("x", fontdict = font2)
    return

# khoi tao main
def Main():

    Thelta =  np.random.randn(2,1)

    for i in range(num_loop):
        h = X_final @ Thelta
        J = Cost_function(h)
        Thelta =  Gradient_descent(Thelta, h)
    Drawing_cost(Thelta_Ones, J_arr)
    Drawing_linear(h)
    plt.show()
    return

# cho chay ham main
Main()
