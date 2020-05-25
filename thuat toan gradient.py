import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def ham_thu(X,theta):
	return np.dot(X,theta)

def gradient(X,y,theta,n):
	loss=ham_thu(X,theta)-y
	return np.dot(loss,X)/n

if __name__=='__main__':
	ds=pd.read_csv('data_train.csv').values
	#ds1=pd.read_csv('C:\\Users\\DELL\\Pictures\\aaa.txt').values
	#khởi tạo x,y,theta
	X=ds[:,:3]
	y=ds[:,3:]
	n,m=X.shape
	theta=np.random.rand(m)

	#khởi tạo mảng để tí nữa tiện vẽ hình
	gradient_0=[]
	gradient_1=[]
	gradient_2=[]

	#tính toán gradient và theta
	count=0
	while count<5000:
		gra=gradient(X,y.squeeze(),theta,n)
		if count<=200:
			gradient_0.append(abs(gra[0]))
			gradient_1.append(abs(gra[1]))
			gradient_2.append(abs(gra[2]))

		theta=theta-0.0001*gra
		count=count+1

	#sau khi tính ta sẽ thử cho vào một mẫu dữ liệu để xem kết quả thu đc là như thế nào. 
	test=[58,93,41]
	print(np.dot(theta,test))

	#vẽ hình mô phỏng
	plt.plot(range(0,201),gradient_0,'g',label='gradient_0')
	plt.plot(range(0,201),gradient_1,'r',label='gradient_1')
	plt.plot(range(0,201),gradient_2,'b',label='gradient_2')
	plt.xlabel('cout')
	plt.ylabel('gradient')
	plt.legend(loc='best')
	plt.show()
