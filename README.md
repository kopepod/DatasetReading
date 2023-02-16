# Fisher Iris Best Model Selection

Selecting the best model to classify the Fisher Iris dataset. This code reads the fisher-iris dataset as csv and selects the best model model via mAp.

## Dependencies

install sk-learn
```bash
pip install scikit-learn
pip install numpy
```

## Code
### Training
```python
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

parser = argparse.ArgumentParser(description="Finding best model for categorical predictions")
parser.add_argument("--File", type = str, required = True, help = "File path")
parser.add_argument("--ModelName", type = str, required = True, help = "Model File Name")

Options = parser.parse_args()

print(Options)

File = Options.File

DF = pandas.read_csv(File)

X = numpy.asarray(DF.iloc[:,0:-1])
Y = DF.iloc[:,-1]
Classes = list(Y.unique())

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
model.Classes = Classes

pickle.dump( model, open( Options.ModelName, "wb" ) )

```
### Testing
```python
#TestBestModel.py
import pickle
import numpy
import argparse

parser = argparse.ArgumentParser(description="Load model to classify categorical samples")
parser.add_argument("--ModelName", type = str, required = True, help = "Model File Name")

Options = parser.parse_args()

print(Options)

x = []

for i in range(4):
	x.append(float(input("x[%d] = " %(i))))

x = numpy.asarray(x)

print("input sample : ", end = "")
print(x)

model = pickle.load( open( Options.ModelName, "rb" ) )

y_hat = model.predict(x.reshape(1,-1))

label = model.Classes[y_hat[0]]

print("Sample predicted class : %s" %(label))


```

# Running
## Training
```bash
python TrainBestModel.py --File https://raw.githubusercontent.com/kopepod/DatasetReading/main/fisher_iris.csv --ModelName BestModel.p
```
## Testing
```bash
python TestBestModel.py --ModelName BestModel.p
```

