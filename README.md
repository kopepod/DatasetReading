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
  Y.iloc[Y == lbl] = i

Y = numpy.asarray(Y)

X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size = 0.8, random_state = 40)

print(X_tr)
print(X_te)

print(Y_tr)
print(Y_te)


```
# Running
```
python Main.py --File https://raw.githubusercontent.com/kopepod/DatasetReading/main/fisher_iris.csv
```


