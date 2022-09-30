# DatasetReading
Reading a dataset with python

This is the code to read a dataset and perform a split using sk-learn

## Dependencies

install sk-learn
```
pip install scikit-learn
```

## Code
```
import numpy
from sklearn.model_selection import train_test_split
import argparse
import pandas

parser = argparse.ArgumentParser(description="Leer con argumentos")
parser.add_argument("--File", type = str, required = True, help = "File path")

Options = parser.parse_args()

print(Options)

File = Options.File

DF = pandas.read_csv(File)

X = numpy.asarray(DF.iloc[:,0:-1])
Y = DF.iloc[:,-1]

for i, lbl in enumerate(Y.unique()):
  Y[:][Y == lbl] = i

Y = numpy.asarray(Y)

X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size = 0.8, random_state = 40)

Y_hat = []

# k-nn
# ------
k = 5

for x_te in X_te:
	d = LA.norm(x_te - X_tr, axis = 1)
	idx = numpy.argsort(d)
	y_hat, _ = scipy.stats.mode(Y_tr[idx[0:k]])
	Y_hat.append(y_hat[0])
# ---
	
Y_hat = numpy.asarray(Y_hat)

acc = numpy.sum(Y_hat == Y_te) / len(Y_hat)


```
# Running
```
python Main.py --File https://raw.githubusercontent.com/kopepod/DatasetReading/main/fisher_iris.csv
```


