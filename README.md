Brightside Data Challenge
-

Hello Brett,

Sean here. First, Thanks for providing the data challenge it was fun.

The route I took with this challenge was to assume that Hardship, Settlement, and Defaults/Charge Offs were all extremly bad for 
the creditor. I created three models that predicted each of those possibilities. I then created a script that evaluated loans and 
only recommened loans that met a set interest rate, not-likely to default, nor settle, nor go through hardship. All three models 
are Random Forest classifiers. All the models were trained and evaluated on undersampled data that had a 50/50 division of the 
majority and minority (hardship, settlement, default) classes. All models attained an accuracy of over 95% on the undersampled 
data. 

I took more of a hack-and-slash method to the cleaning phase of this challenge. Removing all features that included a missing 
value. I corrected some of the skewed features with a logarithm or square root transformation. I made dummies of the categorical 
data.

If you wish to see my thought process I urge you to navigate to the notebook.ipynb file and read through.
If you wish to see a simple way I would apply the models please navigate to main.py.

All of the models are stored in the model directory. The EDA report is saved under report.html.

Thanks!