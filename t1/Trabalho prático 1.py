import numpy as np

data = open('./abalone.data.2', 'r')
arr = np.genfromtxt(data, delimiter=',')
Y_categories = 29
accuracy_means = []
hits_per_cat = np.zeros((Y_categories))
total_per_cat = np.zeros((Y_categories))
accur_per_cat = np.zeros((Y_categories))

i = 0
iters = 100

while i < iters:
  i += 1
  print('iter:', i)

  np.random.shuffle(arr)
  eightyp = int(len(arr) * 0.8)
  training_data = arr[:eightyp][:,:8]
  testing_data = arr[eightyp:][:,:8]

  c = arr[:,8].astype(int)
  v = arr[:,8].astype(int)

  X = training_data.T
  Y = np.zeros((Y_categories,len(X[0])))

  #Y[:, np.arange(Y_categories)] = 1
  for j in range(len(X)): Y[c[j]-1][j] = 1

  A = Y.dot(X.T).dot(np.linalg.pinv(X.dot(X.T)))
  
  accuracy = []
  for l in range(len(testing_data)):
    x = testing_data[l]
    rmax = A.dot(x).argmax(axis=0)
    r = rmax + 1 == v[l]
    accuracy.append(r)
    if r: hits_per_cat[v[l]-1] += 1
    total_per_cat[v[l]-1] += 1

  accuracy_means.append(np.array(accuracy).mean())

print('\n---Resultados---\n\nAcurácia geral:', np.array(accuracy_means).mean())

for i in range(Y_categories):
  t = total_per_cat[i]
  a = hits_per_cat[i]
  p = (a * 100) / t if t > 0 else 0.0
  accur_per_cat[i] = p
  print('Acurácia de categoria %d: %f' % (i+1,p))

print('Desvio padrão:', np.std(accur_per_cat))
