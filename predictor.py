import numpy as np
import pandas as pd
from sklearn import linear_model
#from sklearn.preprocessing import StandardScalar
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import sklearn.metrics as metrics
def getData():
    #allfeatures = ["Age","Height","Wt", "40YD","Vertical","BenchReps","Broad_Jump", "3Cone", "Shuttle","Class","G_season","Rec_Season", "Rec_Yds_Season", "Rec_Avg_Season", "Rec_TD_Season", "Career_Rec_Yds", "Career_Rec_Avg", "Career_Rec_TD"]
 
    features = ["Age","Height","Wt", "40YD","Vertical","BenchReps","Broad_Jump", "3Cone", "Shuttle","Class","G_season","Rec_Season", "Rec_Yds_Season", "Rec_Avg_Season", "Rec_TD_Season", "Career_Rec_Yds", "Career_Rec_Avg", "Career_Rec_TD"]
 
    dep = ["Average_AV"]
    #dep = ["AAV2"]
    data = pd.read_csv("data.csv")
    #print(data)
    #data = data[data.Average_AV != 0]
    dtrain = data[data.Year < 2018]
    df = pd.DataFrame(dtrain, columns=features)
    df.fillna(0, inplace=True)
    #df.fillna(df.mean(), inplace=True)
 
    target = pd.DataFrame(dtrain, columns=dep)
    dtest = data[data.Year >= 2018]
    dftest = pd.DataFrame(dtest, columns=features)
    dftest.fillna(0, inplace=True)
    #df.fillna(df.mean(), inplace=True)
 
    targettest = pd.DataFrame(dtest, columns=dep)
    
    return dtrain,df,target, dftest, targettest, dtest

def makeModel(X,y):
    lm = linear_model.LinearRegression()
    model = lm.fit(X,y)
    #predictions = lm.predict(X)
    #print(predictions[0:5])
    #print("score", lm.score(X,y))
    return lm, lm.score(X,y)

def collinearityCheck(data):
    corr = data.corr()
    heatmap = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True)
    fig = heatmap.get_figure()
    fig.savefig("heatmap")

def normalizeData(data):
    norm_data = (data - data.mean())/data.std()
    #print(norm_data)
    return norm_data

def minMaxNormalizeData(data):
    norm_data = (data - data.min())/(data.max() - data.min())
    #print(norm_data)
    return norm_data

def powerset(s):
    x = len(s)
    power = []
    for i in range(1, 1 << x):
        power.append([s[j] for j in range(x) if (i & ( 1 << j))])
    print(x, len(power))
    return power


def testAll(X,y):
    features = ["Age","Height","Wt", "40YD","Vertical","BenchReps","Broad_Jump", "3Cone", "Shuttle","Class","G_season","Rec_Season", "Rec_Yds_Season", "Rec_Avg_Season", "Rec_TD_Season", "Career_Rec_Yds", "Career_Rec_Avg", "Career_Rec_TD"]
    value = []
    allfeatures = powerset(features)
    #allfeatures = allfeatures[-5000:]
    for f in allfeatures:
        df = pd.DataFrame(X, columns=f)
        #print(df)
        df.fillna(0, inplace=True)
        #print("*"*70)
        #print(f)
        lm,score = makeModel(df, y)
        value.append(score)
    sortVal = sorted(zip(value,allfeatures),reverse=True)
    print(sortVal[0])
    #best was all features used
 
def summary(X,y,lm, players):
    predicted = lm.predict(X)
    explained_variance = metrics.explained_variance_score(y, predicted)
    mean_absolute_error = metrics.mean_absolute_error(y, predicted)
    mse = metrics.mean_squared_error(y, predicted)
    mean_squared_log_error = metrics.mean_squared_log_error(y, predicted)
    median_absolute_error = metrics.median_absolute_error(y, predicted)
    r2 = metrics.r2_score(y, predicted)
    ynum = y.to_numpy()
    playnum = players.to_numpy()
    max_diff = np.abs(ynum[0]-predicted[0])
    min_diff = max_diff
    max_index = 0
    min_index = 0

    for i in range(0,len(predicted)):
        diff = np.abs(ynum[i] - predicted[i])
        if diff > max_diff:
            max_diff = diff
            max_index = i
        elif diff < min_diff:
            min_diff = diff
            min_index = i

    print("*"*30,"Summary","*"*30)
    print(list(X))
    print("coef:",lm.coef_)
    print("intercept:", lm.intercept_)
    print("explained_variance:", explained_variance)
    print("mean_squared_log_error:", mean_squared_log_error)
    print("median_absolute_error:", median_absolute_error)
    print("r2:", r2)
    print("MAE:", mean_absolute_error)
    print("MSE:", mse)
    print("RMSE:", np.sqrt(mse))
    print("Max Diff:", max_diff, "at:", playnum[max_index])
    print("Min Diff:", min_diff, "at:", playnum[min_index])
    print("Predicted Max:", max(predicted))
    print("Predicted Min:", min(predicted))
    print("Predicted Avg:", np.mean(predicted))
    rankPlayers(playnum, predicted, ynum, 10)
    print("*"*69)

def rankPlayers(players, pred, true, count):
    psort = sorted(zip(pred, players, true), reverse=True)
    psortbad = sorted(zip(pred, players, true))
    print("Top", count)
    for i in range(count):
        print(i+1,"-",psort[i])
    print("Bottom", count)
    for i in range(count):
        print(i+1,"-",psortbad[i])

def polyModel(X,y,players, Xt, yt, pt):
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    print("indep vars", len(X_poly[0]), "used to be:", len(X[0]))
    pol_lin = linear_model.LinearRegression()
    pol_lin.fit(X_poly, y)
    predictions = pol_lin.predict(X_poly)
    print(predictions[0:5])
    print("score", pol_lin.score(X_poly,y))
    summary(X_poly,y,pol_lin, players)
    Xt_poly = poly.fit_transform(Xt)
    summary(Xt_poly,yt,pol_lin,pt)

if __name__=='__main__':
    data,X,y,Xt,yt,dtest = getData()
    #print(data)
    #print(data.dtypes)
    #print(X)
    #print(y)
    players = pd.DataFrame(data, columns=["Player"])
    ptest = pd.DataFrame(dtest, columns=["Player"])
    collinearityCheck(data)
    normX = normalizeData(X)
    mmX = minMaxNormalizeData(X)
    ny = normalizeData(y)
    mmy = minMaxNormalizeData(y)
    lm,score = makeModel(X,y)
    print("*"*70)
    print("regular")
    print(lm.coef_)
    print(lm.intercept_)
    print(score)
    print("*"*70)
    print("norm")
    lmn,score = makeModel(normX, y)
    print(lmn.coef_)
    print(lm.intercept_)
    print(score)
    lmm,score = makeModel(mmX, y)
    print("*"*70)
    print("minMax")
    print(lmm.coef_)
    print(lmm.intercept_)
    print(score)
    print("*"*70)
    print("norm y")
    lmn,score = makeModel(normX, ny)
    print(lmn.coef_)
    print(lm.intercept_)
    print(score)
    lmm,score = makeModel(mmX, mmy)
    print("*"*70)
    print("minMax")
    print(lmm.coef_)
    print(lmm.intercept_)
    print(score)
    print("*"*70)
    #testAll(data, y)
    summary(X,y,lm, players)
    print("Testing")
    summary(Xt, yt, lm, ptest)
    print("*"*30,"Testing Poly","*"*30)
    Xnp = X.values
    Xtnp = Xt.values
    ynp = y.values
    #polyModel(Xnp, y, players,Xt,yt,ptest)


