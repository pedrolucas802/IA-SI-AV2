import numpy as np
import matplotlib.pyplot as plt
from random import random as rd
np.random.seed(0)

n_samples = 1200
mean1 = [rd()+2, rd()-2]
cov1 = [[rd()+1, rd()-2.8], [rd()-0.8, rd()+1]]
d1 = np.random.multivariate_normal(mean1, cov1, n_samples)
r1 = np.ones((n_samples,1))

mean3 = [0, rd()-2]
cov3 = [[rd()-0.2, rd()+0.2], [rd()+0.2, rd()-0.3]]
d2n = np.random.multivariate_normal(mean3, cov3, int(n_samples*.25))-0.9
r2n = +np.ones((int(n_samples*.25),1))


mean2 = [rd()-4, rd()-4]
cov2 = [[rd()+1, rd()-0.8], [rd()-0.8, rd()+1]]
d2 = np.random.multivariate_normal(mean2, cov2, n_samples)-2.2
r2 = -np.ones((n_samples,1))


mean3 = [0, 0]
cov3 = [[0.2, rd()-0.2], [rd()+0.2,-0.4]]
d1n = np.random.multivariate_normal(mean3, cov3, int(n_samples*.55))-2.9
r1n = -np.ones((int(n_samples*.55),1))


X = np.concatenate((d1,d2n,d2,d1n))
Y = np.concatenate((r1,r2n,r2,r1n))
Data = np.array(np.concatenate((X,Y),axis=1))
np.savetxt("DataAV2.csv",Data,delimiter=',')

plt.show()