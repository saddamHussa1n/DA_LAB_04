import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import time
start_time = time.time()


def amount_of_errors(arr1, arr2):
    count = 0
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            count += 1
    return count


def linear_regression(x_train, y_train, x_test):
    Y = np.zeros((len(x_train.index), 3))
    i = 0
    for item in y_train:
        Y[i][item - 1] = 1
        i += 1

    X = np.array(x_train)
    X = np.c_[np.ones(len(x_train.index)), X]
    Xt = X.transpose()
    buf = Xt.dot(X)
    inv = np.linalg.inv(buf)
    buf = inv.dot(Xt)
    B = buf.dot(Y)

    x = np.array(x_test)
    x = np.c_[np.ones(len(x_test.index)), x]
    est = x.dot(B)

    result = []
    for row in est:
        r = list(row)
        n = r.index(max(r)) + 1
        result.append(n)
    return result


def knn(X_train, y_train, X_test):
    neighbors_3_res = []
    neighbors_5_res = []
    X = np.array(X_train)
    x = np.array(X_test)
    y = np.array(y_train)
    for i in range(len(X_test.index)):
        dtype = [('dist', float), ('class', int)]
        distances = np.array([], dtype=dtype)
        for j in range(len(X_train.index)):
            dist = 0
            for k in range(len(X_train.columns)):
                dist += (x[i][k] - X[j][k]) ** 2
            dist = dist ** 0.5
            distances = np.append(distances, np.array([(dist, y[j])], dtype=distances.dtype))
        distances = np.sort(distances, order='dist')
        classes_3 = np.array([distances[0][1], distances[1][1], distances[2][1]])
        classes_5 = np.array([distances[0][1], distances[1][1], distances[2][1], distances[3][1], distances[4][1]])
        neighbors_3_res.append(np.bincount(classes_3).argmax())
        neighbors_5_res.append(np.bincount(classes_5).argmax())
    return neighbors_3_res, neighbors_5_res


def linear_discriminant(df, X_test):
    result = []
    x = np.array(X_test)

    for i in range(15):
        set1 = df[df['class'] == 1]
        pi1 = len(set1.index)
        mu1 = np.array([set1['x.1'].mean(), set1['x.2'].mean(), set1['x.3'].mean(), set1['x.4'].mean()])
        X1 = set1[['x.1', 'x.2', 'x.3', 'x.4']]

        set2 = df[df['class'] == 2]
        pi2 = len(set2.index)
        mu2 = np.array([set2['x.1'].mean(), set2['x.2'].mean(), set2['x.3'].mean(), set2['x.4'].mean()])
        X2 = set2[['x.1', 'x.2', 'x.3', 'x.4']]

        set3 = df[df['class'] == 3]
        pi3 = len(set3.index)
        mu3 = np.array([set3['x.1'].mean(), set3['x.2'].mean(), set3['x.3'].mean(), set3['x.4'].mean()])
        X3 = set3[['x.1', 'x.2', 'x.3', 'x.4']]

        sigma = np.zeros((4, 4))
        X1 = np.array(X1)
        X2 = np.array(X2)
        X3 = np.array(X3)

        for j in range(3):
            if j == 0:
                for k in range(pi1):
                    M = X1[k] - mu1
                    M = M.reshape(4, 1)
                    matr = M.dot(M.transpose())
                    sigma += matr
            if j == 1:
                for k in range(pi2):
                    M = X2[k] - mu2
                    M = M.reshape(4, 1)
                    matr = M.dot(M.transpose())
                    sigma += matr
            if j == 2:
                for k in range(pi3):
                    M = X3[k] - mu3
                    M = M.reshape(4, 1)
                    matr = M.dot(M.transpose())
                    sigma += matr
        sigma = sigma * (1 / 132)

        xi = np.array(x[i])
        xi = xi.reshape(4, 1)
        mu1 = mu1.reshape(4, 1)
        mu2 = mu2.reshape(4, 1)
        mu3 = mu3.reshape(4, 1)
        delta1 = (xi.transpose().dot(np.linalg.inv(sigma))).dot(mu1) - (
                0.5 * mu1.transpose().dot(np.linalg.inv(sigma))).dot(mu1) + np.log(pi1)
        delta2 = (xi.transpose().dot(np.linalg.inv(sigma))).dot(mu2) - (
                0.5 * mu2.transpose().dot(np.linalg.inv(sigma))).dot(mu2) + np.log(pi2)
        delta3 = (xi.transpose().dot(np.linalg.inv(sigma))).dot(mu3) - (
                0.5 * mu3.transpose().dot(np.linalg.inv(sigma))).dot(mu3) + np.log(pi3)

        buf = [delta1, delta2, delta3]
        n = buf.index(max(buf)) + 1
        result.append(n)
    return result


def quadratic_discriminant(df, X_test):
    result = []
    x = np.array(X_test)

    for i in range(15):
        set1 = df[df['class'] == 1]
        pi1 = len(set1.index)
        mu1 = np.array([set1['x.1'].mean(), set1['x.2'].mean(), set1['x.3'].mean(), set1['x.4'].mean()])
        X1 = set1[['x.1', 'x.2', 'x.3', 'x.4']]

        set2 = df[df['class'] == 2]
        pi2 = len(set2.index)
        mu2 = np.array([set2['x.1'].mean(), set2['x.2'].mean(), set2['x.3'].mean(), set2['x.4'].mean()])
        X2 = set2[['x.1', 'x.2', 'x.3', 'x.4']]

        set3 = df[df['class'] == 3]
        pi3 = len(set3.index)
        mu3 = np.array([set3['x.1'].mean(), set3['x.2'].mean(), set3['x.3'].mean(), set3['x.4'].mean()])
        X3 = set3[['x.1', 'x.2', 'x.3', 'x.4']]

        sigma1 = np.array(X1.cov())
        sigma2 = np.array(X2.cov())
        sigma3 = np.array(X3.cov())

        xi = np.array(x[i])
        xi = xi.reshape(4, 1)
        mu1 = mu1.reshape(4, 1)
        mu2 = mu2.reshape(4, 1)
        mu3 = mu3.reshape(4, 1)

        delta1 = -0.5 * np.log(np.linalg.det(sigma1)) - 0.5 * (xi - mu1).transpose().dot(np.linalg.inv(sigma1)).dot(
            xi - mu1) + np.log(pi1)
        delta2 = -0.5 * np.log(np.linalg.det(sigma2)) - 0.5 * (xi - mu2).transpose().dot(np.linalg.inv(sigma2)).dot(
            xi - mu2) + np.log(pi2)
        delta3 = -0.5 * np.log(np.linalg.det(sigma3)) - 0.5 * (xi - mu3).transpose().dot(np.linalg.inv(sigma3)).dot(
            xi - mu3) + np.log(pi3)

        buf = [delta1, delta2, delta3]
        n = buf.index(max(buf)) + 1
        result.append(n)
    return result


# 1
dataset = pd.read_csv("var11.csv", sep=';')

for col in dataset:
    if col != 'class':
        dataset[col] = [x.replace(',', '.') for x in dataset[col]]
dataset['x.1'] = dataset['x.1'].astype(float)
dataset['x.2'] = dataset['x.2'].astype(float)
dataset['x.3'] = dataset['x.3'].astype(float)
dataset['x.4'] = dataset['x.4'].astype(float)

# 2
test_sample = dataset.sample(n=15)
df = dataset.drop(test_sample.index)

# 3
X_train = df[['x.1', 'x.2', 'x.3', 'x.4']]
y_train = df['class']
X_test = test_sample[['x.1', 'x.2', 'x.3', 'x.4']]
y_test = test_sample['class']

res = linear_regression(X_train, y_train, X_test)
print("-------------------3-------------------")
print("Истинные значения:")
print(list(y_test))
print()
print("Прогнозы (лин регр модель):")
print(res)
print("Ошибки:", amount_of_errors(list(y_test), res))
print()

# 4
knn_3, knn_5 = knn(X_train, y_train, X_test)
print("-------------------4-------------------")
print("Прогнозы (мой knn с k=3):")
print(knn_3)
print("Ошибки:", amount_of_errors(list(y_test), knn_3))
print()
print("Прогнозы (мой knn с k=5):")
print(knn_5)
print("Ошибки:", amount_of_errors(list(y_test), knn_5))
print()

k_clf = KNeighborsClassifier(n_neighbors=3)
k_clf.fit(X_train, y_train)
prediction = k_clf.predict(X_test)

print("Прогнозы (knn с k=3):")
print(list(prediction))
print("Ошибки:", amount_of_errors(list(y_test), prediction))
print()

k_clf = KNeighborsClassifier(n_neighbors=5)
k_clf.fit(X_train, y_train)
prediction = k_clf.predict(X_test)

print("Прогнозы (knn с k=5):")
print(list(prediction))
print("Ошибки:", amount_of_errors(list(y_test), prediction))
print()

# 5
sns.scatterplot(data=dataset, x=dataset['x.1'], y=dataset['x.2'], hue=dataset['class'], style=dataset['class'],
                palette='deep')
plt.show()

# 7
X = pd.concat([df['x.1'], df['x.2']], axis=1)
min1, max1 = X['x.1'].min() - 1, X['x.1'].max() + 1
min2, max2 = X['x.2'].min() - 1, X['x.2'].max() + 1
x1grid = np.linspace(min1, max1, 50)
x2grid = np.linspace(min2, max2, 50)
xx, yy = np.meshgrid(x1grid, x2grid)
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
grid = np.hstack((r1, r2))

grid = pd.DataFrame(data=grid)

knn_3, knn_5 = knn(X, y_train, grid)
zz = np.array(knn_3).reshape(xx.shape)

plt.contourf(xx, yy, zz, cmap='Paired')
sns.scatterplot(data=dataset, x=dataset['x.1'], y=dataset['x.2'], hue=dataset['class'], style=dataset['class'],
                palette='deep')
plt.show()

zz = np.array(knn_5).reshape(xx.shape)

plt.contourf(xx, yy, zz, cmap='Paired')
sns.scatterplot(data=dataset, x=dataset['x.1'], y=dataset['x.2'], hue=dataset['class'], style=dataset['class'],
                palette='deep')
plt.show()

# 6
res = linear_regression(X, y_train, grid)

zz = np.array(res).reshape(xx.shape)
plt.contourf(xx, yy, zz, cmap='Paired')
sns.scatterplot(data=dataset, x=dataset['x.1'], y=dataset['x.2'], hue=dataset['class'], style=dataset['class'],
                palette='deep')
plt.show()

# 8
print("-------------------8-------------------")
result = linear_discriminant(df, X_test)
print("Прогнозы (линейный дискриминантный анализ):")
print(result)
print("Errors:", amount_of_errors(list(y_test), result))
print()

# 9
print("-------------------9-------------------")
result = quadratic_discriminant(df, X_test)
print("Прогнозы (квадратичный дискриминантный анализ):")
print(result)
print("Ошибки:", amount_of_errors(list(y_test), result))
print()

# 10
data = dataset[(dataset['class'] == 1) | (dataset['class'] == 2)]
set = df[(df['class'] == 1) | (df['class'] == 2)]
test = test_sample[(test_sample['class'] == 1) | (test_sample['class'] == 2)]

X_train = set[['x.1', 'x.2', 'x.3', 'x.4']]
y_train = set['class']
X_test = test[['x.1', 'x.2', 'x.3', 'x.4']]
y_test = test['class']

svc_clf = SVC(kernel='linear')
svc_clf.fit(X_train, y_train)
prediction = svc_clf.predict(X_test)

print("-------------------10-------------------")
print("Истинные значения:")
print(list(y_test), '\n')
print("Пронозы (SVM-метод):")
print(list(prediction))
print("Ошибки:", amount_of_errors(list(y_test), prediction), '\n\n')

X = pd.concat([set['x.1'], set['x.2']], axis=1)
min1, max1 = X['x.1'].min() - 1, X['x.1'].max() + 1
min2, max2 = X['x.2'].min() - 1, X['x.2'].max() + 1
x1grid = np.linspace(min1, max1, 100)
x2grid = np.linspace(min2, max2, 100)
xx, yy = np.meshgrid(x1grid, x2grid)
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
grid = np.hstack((r1, r2))

model = SVC(kernel='linear')
model.fit(X, y_train)
yhat = model.predict(grid)
zz = yhat.reshape(xx.shape)

plt.contourf(xx, yy, zz, cmap='Paired')
sns.scatterplot(data=data, x=df['x.1'], y=data['x.2'], hue=data['class'], style=data['class'], palette='deep')
plt.show()

# 11
lr_errors = np.array([])
knn_3_errors = np.array([])
knn_5_errors = np.array([])
lda_errors = np.array([])
qda_errors = np.array([])

for i in range(100):
    test_sample = dataset.sample(n=15)
    df = dataset.drop(test_sample.index)

    X_train = df[['x.1', 'x.2', 'x.3', 'x.4']]
    y_train = df['class']
    X_test = test_sample[['x.1', 'x.2', 'x.3', 'x.4']]
    y_test = test_sample['class']

    res = linear_regression(X_train, y_train, X_test)
    num = amount_of_errors(list(y_test), res)
    lr_errors = np.append(lr_errors, num / 15)

    res1, res2 = knn(X_train, y_train, X_test)
    num = amount_of_errors(list(y_test), res1)
    knn_3_errors = np.append(knn_3_errors, num / 15)
    num = amount_of_errors(list(y_test), res2)
    knn_5_errors = np.append(knn_5_errors, num / 15)

    res = linear_discriminant(df, X_test)
    num = amount_of_errors(list(y_test), res)
    lda_errors = np.append(lda_errors, num / 15)

    res = quadratic_discriminant(df, X_test)
    num = amount_of_errors(list(y_test), res)
    qda_errors = np.append(qda_errors, num / 15)

table = pd.DataFrame({'Error': [np.mean(lr_errors), np.mean(knn_3_errors), np.mean(knn_5_errors), np.mean(lda_errors),
                                np.mean(qda_errors)]}, index=['LR', 'KNN_3', 'KNN_5', 'LDA', 'QDA'])
print("-------------------11-------------------")
print(table, '\n')
print("--- %s seconds ---" % (time.time() - start_time))
