import numpy as np

def distance_matrix(data, dimension, delay, norm):
    N = int(len(data) - (dimension-1) * delay)
    distance_matrix = np.zeros((N, N), dtype="float32")
    if norm == 'manhattan':     
        for i in range(N):
            for j in range(i, N, 1):
                temp = 0.0
                for k in range(dimension):
                    temp += np.abs(data[i+k*delay] - data[j+k*delay])
                distance_matrix[i,j] = distance_matrix[j,i] = temp
    elif norm == 'euclidean':
        for i in range(N):
            for j in range(i, N, 1):
                temp = 0.0
                for k in range(dimension):
                    temp += np.power(data[i+k*delay] - data[j+k*delay], 2)
                distance_matrix[i,j] = distance_matrix[j,i] = np.sqrt(temp)
    elif norm == 'supremum':
        temp = np.zeros(dimension)
        for i in range(N):
            for j in range(i, N, 1):
                for k in range(dimension):
                    temp[k] = np.abs(data[i+k*delay] - data[j+k*delay])
                distance_matrix[i,j] = distance_matrix[j,i] = np.max(temp)
    return distance_matrix

def recurrence_matrix(data, dimension, delay, threshold, norm):
    N = int(len(data) - (dimension-1) * delay)
    distance_matrix_1 = distance_matrix(data, dimension, delay, norm)
    for i in range(N):
        for j in range(i, N, 1):
            if distance_matrix_1[i,j] <= threshold:
                distance_matrix_1[i,j] = distance_matrix_1[j,i] = 1
            else:
                distance_matrix_1[i,j] = distance_matrix_1[j,i] = 0
    return distance_matrix_1.astype(int)

def joint_matrix(data1, data2, dimension1, dimension2, delay1, delay2, threshold1, threshold2, norm1, norm2):
    N1 = int(len(data1) - (dimension1-1) * delay1)
    N2 = int(len(data2) - (dimension2-1) * delay2)
    assert N1 == N2, "Space phase must have the same size"
    recurrence_matrix_1 = recurrence_matrix(data1, dimension1, delay1, threshold1, norm1)
    recurrence_matrix_2 = recurrence_matrix(data2, dimension2, delay2, threshold2, norm2)
    for i in range(N1):
        for j in range(i, N1, 1):
            if recurrence_matrix_1[i,j] == 1 and recurrence_matrix_2[i,j] == 1:
                recurrence_matrix_1[i,j] = recurrence_matrix_1[j,i] = 1
            else:
                recurrence_matrix_1[i,j] = recurrence_matrix_1[j,i] = 0
    return recurrence_matrix_1.astype(int)


import pandas as pd
import matplotlib.pyplot as plt

#x_solution
data = pd.read_csv(r"C:\Users\hp\Desktop\Prof Sujith Research Lab\x_rossler_extra.csv")
ser1 = data.iloc[0].reset_index(drop=True).squeeze()
x_patterns = data.values.tolist()
array_x = np.asarray(x_patterns)
print(type(array_x))


#y_solution
data2 = pd.read_csv(r"C:\Users\hp\Desktop\Prof Sujith Research Lab\rossler_y_extra.csv")
ser2 = data2.iloc[0].reset_index(drop=True).squeeze()
y_patterns = data2.values.tolist()
array_y = np.asarray(y_patterns)
print(type(array_y))

#z_solution
data3 = pd.read_csv(r"C:\Users\hp\Desktop\Prof Sujith Research Lab\rossler_z_extra.csv")
ser3 = data3.iloc[0].reset_index(drop=True).squeeze()
z_patterns = data3.values.tolist()
array_z = np.asarray(z_patterns)
print(type(array_z))

#getting the arrays
array_x_s = array_x[0:400]
array_y_s = array_y[0:400]
array_z_s = array_z[0:200]
random = np.random.randn(1000)
cross_rec = joint_matrix(array_x_s,array_y_s,1,1,1,1,0.1,0.1,'euclidean','euclidean')
plt.figure(figsize = (6,6))
plt.imshow(cross_rec, cmap = 'binary', origin = 'lower')
plt.title(' Cross Recurrence Plot for rossler x and y')
plt.xlabel('Time')
plt.ylabel('Time')
plt.show()

