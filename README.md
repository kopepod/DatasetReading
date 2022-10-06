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
# TrainBestModel.py
import pickle
import numpy
from sklearn.model_selection import train_test_split
import argparse
import pandas
from numpy import linalg as LA
import scipy
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Leer con argumentos")
parser.add_argument("--File", type = str, required = True, help = "File path")
parser.add_argument("--ModelName", type = str, required = True, help = "Model File Name")

Options = parser.parse_args()

print(Options)

File = Options.File

DF = pandas.read_csv(File)

X = numpy.asarray(DF.iloc[:,0:-1])
Y = DF.iloc[:,-1]

for i, lbl in enumerate(Y.unique()):
  Y[:][Y == lbl] = i

Y = numpy.asarray(Y)
Y = Y.astype('int')

Models = {
"SVC" : SVC(),
"GNB": GaussianNB(),
"KNN": KNeighborsClassifier(),
"DTree" : DecisionTreeClassifier(),
"RF" : RandomForestClassifier()
}

Seeds = [0,10,40,120,400,320,570,999,1240,9999]

Result = []

for name, model in Models.items():
	print("Testing %s " %(model))
	for seed in tqdm(Seeds):
		X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size = 0.2, random_state = seed)
		model.fit(X_tr,Y_tr)
		Y_hat = model.predict(X_te)
		acc = numpy.sum(Y_hat == Y_te)/len(Y_hat)
		Result.append([name, seed, acc])

Result = pandas.DataFrame(Result, columns = ["Model","Seed","Acc"])
print(Result)

Lista = []

for name, model in Models.items():
	sDF = Result[Result.Model == name]
	Lista.append([name,sDF.Acc.mean()])
	print("Model %s : mAp %0.3f" %(name, sDF.Acc.mean()) )


Lista = pandas.DataFrame(Lista, columns = ["Model","mAp"])

Lista = Lista.sort_values(by = "mAp", ascending = False);

Lista = Lista.reset_index(drop=True)

BestModel = Lista.at[0,"Model"];

print("Best model %s" %(BestModel) )

model = Models.get(BestModel)

X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size = 0.2, random_state = 0)
model.fit(X_tr,Y_tr)

pickle.dump( model, open( Options.ModelName, "wb" ) )


```
# Running
```
python Main.py --File https://raw.githubusercontent.com/kopepod/DatasetReading/main/fisher_iris.csv
```


