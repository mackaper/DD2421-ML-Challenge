import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
# load the data
trainData = pd.read_csv('TrainOnMe.csv')
evalData = pd.read_csv('EvaluateOnMe.csv')

tdataModify = trainData.copy()
# remove the first column since it is just an index
tdataModify = tdataModify.drop(columns=tdataModify.columns[0])

#same with eval data
edataModify = evalData.copy()
edataModify = edataModify.drop(columns=edataModify.columns[0])

# Ignore this:
# print the data types of the columns to understand the values
# print(trainData.dtypes)
# we se that y and x7 are objects and x12 is boolean, plot to investigate 
# plt.hist(trainData.y)
# y is company name, it is not continous
# now check x7:
# plt.hist(trainData.x7)
# print(np.unique(trainData.x7))
# output: ['AI' 'EBIT/Wh' 'Q1' 'Q2' 'Q3']
# now check x12:
# plt.hist(trainData.x12)
# plt.show()

# drop the columns we dont need, y is the target and x12 is boolean
Xfeatures = tdataModify.drop(columns=['y', 'x12'])
Ytarget = tdataModify['y']
#print(Ytarget)

XfeaturesEval = edataModify.drop(columns='x12')


# check the unique values of the object columns
categoricalColumns = Xfeatures.select_dtypes(include=['object', 'bool']).columns

#print(categoricalColumns)
# we see from the print that x7 is the only categorical column, so we only need to get dummies for x7

# get dummies for the columns
Xdummies = pd.get_dummies(Xfeatures, columns=categoricalColumns)
XdummiesEval = pd.get_dummies(XfeaturesEval, columns=categoricalColumns)
#print(Xdummies.head())

# align the dataframes to ensure they have the same columns
Xdummies, XdummiesEval = Xdummies.align(XdummiesEval, join='left', axis=1, fill_value=0)
#print(Xdummies.head())

# encode the target column
encoder = LabelEncoder()
YtargetEncoded = encoder.fit_transform(Ytarget)
#print(YtargetEncoded)

print(f"training data shape (encoded): {Xdummies.shape}")
print(f"evaluation data shape (encoded): {XdummiesEval.shape}")

# now train the model
# random forest classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=7)
#cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True)

# store the accuracies of each fold
accuracies = []

# loop through each fold
for fold, (trainIdx, evalIdx) in enumerate(cv.split(Xdummies, YtargetEncoded), 1):

    #step 1, split data
    Xtrain, Xeval = Xdummies.iloc[trainIdx], Xdummies.iloc[evalIdx]
    Ytrain, Yeval = YtargetEncoded[trainIdx], YtargetEncoded[evalIdx]
    print(f"training shape: {Xtrain.shape}, eval shape: {Xeval.shape}")


    #step 2, scale features
    scaler = StandardScaler()
    Xtrain3 = scaler.fit_transform(Xtrain)
    Xeval3 = scaler.transform(Xeval)

    #step 3, PCA, linear transformation on the data
    pca = PCA()
    Xtrain4 = pca.fit_transform(Xtrain3)
    Xeval4 = pca.transform(Xeval3)

    print(f"training shape after PCA: {Xtrain4.shape}, eval shape after PCA: {Xtrain4.shape}")

    #train 
    rf.fit(Xtrain4, Ytrain)

    #predict
    Ypredict = rf.predict(Xeval4)

    #accuracy
    accuracy = accuracy_score(Yeval, Ypredict)
    accuracies.append(accuracy)
    print(f"fold {fold} accuracy: {accuracy}")

print("fold accuracies:")
print(accuracies)
print(f"avg accuracy across the folds: {np.mean(accuracies):.4f}")

# train the model on the whole dataset before final prediction

XfinalScaled = scaler.fit_transform(Xdummies)
XfinalPCA = pca.fit_transform(XfinalScaled)

rf.fit(XfinalPCA, YtargetEncoded)

# testPred = rf.predict(XfinalPCA)
# acc = accuracy_score(YtargetEncoded, testPred)
# print(f"accuracy on training data: {acc}")

#now for the eval data
XdummiesEvalScaled = scaler.transform(XdummiesEval)
XdummiesEvalPCA = pca.transform(XdummiesEvalScaled)

YevalPredictEncoded = rf.predict(XdummiesEvalPCA)
#back to the company names
YevalPredict = encoder.inverse_transform(YevalPredictEncoded)

#save to text file
with open('predictions.txt', 'w') as f:
    for label in YevalPredict:
        f.write(f"{label}\n")

print("predicitions saved to 'predictions.txt'")
