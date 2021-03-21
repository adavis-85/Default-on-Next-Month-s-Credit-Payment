using Pkg
Pkg.add("GLM")
using GLM
Pkg.add("StatsBase")
using StatsBase
Pkg.add("Lathe")
using Lathe
Pkg.add("MLBase")
using MLBase
Pkg.add("ClassImbalance")
using ClassImbalance
Pkg.add("ROCAnalysis")
using ROCAnalysis
Pkg.add("DataFrames")
using DataFrames
using Pkg
Pkg.add("Flux")
using Flux
Pkg.add("CSV")
using CSV

##Change to the specific path that you download this from

a=CSV.read("##UCI_Credit_Card.csv")

using Lathe.preprocess: TrainTestSplit

##One hot for all of the columns that are assigned a variable or rating.
Lathe.preprocess.OneHotEncode(a,:SEX)
Lathe.preprocess.OneHotEncode(a,:EDUCATION)
Lathe.preprocess.OneHotEncode(a,:MARRIAGE)
Lathe.preprocess.OneHotEncode(a,:PAY_0)
Lathe.preprocess.OneHotEncode(a,:PAY_2)
Lathe.preprocess.OneHotEncode(a,:PAY_3)
Lathe.preprocess.OneHotEncode(a,:PAY_4)
Lathe.preprocess.OneHotEncode(a,:PAY_5)
Lathe.preprocess.OneHotEncode(a,:PAY_6)



##Splitting the data into the training set at 75% of data and 25% for testing.
train, test = TrainTestSplit(a,.75)

##Fitting all of the columns to a general linear model for linear regression.
fm = @formula(default_payment_next_month~LIMIT_BAL                 
 +SEX                       
 +EDUCATION                 
 +MARRIAGE                  
 +AGE                       
 +PAY_0                     
 +PAY_2                     
 +PAY_3                     
 +PAY_4                     
 +PAY_5                     
 +PAY_6                     
 +BILL_AMT1                 
 +BILL_AMT2                 
 +BILL_AMT3                 
 +BILL_AMT4                 
 +BILL_AMT5                 
 +BILL_AMT6                 
 +PAY_AMT1                  
 +PAY_AMT2                  
 +PAY_AMT3                  
 +PAY_AMT4                  
 +PAY_AMT5                  
 +PAY_AMT6 )
logit = glm(fm,train,Binomial(), ProbitLink())

##Prediction functions.  
prediction = predict(logit,test)
zero_or_one = [if x < 0.5 0 else 1 end for x in prediction]
    
PREDICTION_DATA = DataFrame(y_train = test.default_payment_next_month, y_test = zero_or_one, prob_predicted = prediction);
PREDICTION_DATA.correctly_classified = PREDICTION_DATA.y_train .== PREDICTION_DATA.y_test
accur = mean(PREDICTION_DATA.correctly_classified)

##Taking in the information
new=DataFrame(train[2,:])
predict_for_show=predict(logit,new)
if predict_for_show[1][1]<.5
    println(0)
else
    println(1)
end

##Check to see if predicted accurately
train.default_payment_next_month[2]

##Using the most statistically significant columns fitted to a linear regression
fm = @formula(default_payment_next_month~LIMIT_BAL                 
 +SEX                       
 +EDUCATION                 
 +MARRIAGE                  
 +AGE                       
 +PAY_0                     
 +PAY_2                     
 +PAY_3                                          
 +BILL_AMT1                                  
 +BILL_AMT6                 
 +PAY_AMT1                  
 +PAY_AMT2                                    
 +PAY_AMT4                  
  )
logit = glm(fm,train,Binomial(), ProbitLink())

prediction = predict(logit,test)

zero_or_one = [if x < 0.5 0 else 1 end for x in prediction]
    
PREDICTION_DATA = DataFrame(y_train = test.default_payment_next_month, y_test = zero_or_one, prob_predicted = prediction);
PREDICTION_DATA.correctly_classified = PREDICTION_DATA.y_train .== PREDICTION_DATA.y_test
accur = mean(PREDICTION_DATA.correctly_classified)

confusion_matrix = MLBase.roc(PREDICTION_DATA.y_train, PREDICTION_DATA.y_test)


