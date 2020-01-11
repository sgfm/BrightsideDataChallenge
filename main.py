import pandas as pd
import numpy as np
import pickle
import math

def check_interest(loan, desired_interest):
    '''
    Checks if the loan has the desired interest rate to invest. Returns boolean.
    '''
    return float(loan.int_rate[:-1]) >= desired_interest

def format_loan(loan):
    '''
    This takes in a loan and formats it to be evaluated by the machine learning models. This is a replication of the cleaning process seen in the notebook.ipynb
    '''
    #Loading in the columns
    col = pickle.load( open( "columns.pkl", "rb" ) )
    zero_data = np.zeros(shape=(1,len(col)))
    #Creating the zeroed dataframe 
    s = pd.DataFrame(zero_data, columns=col)
    #Populating the dataframe with the correct dummies from the loan
    dummy_lis = ['sub_grade', 'home_ownership', 'verification_status', 'issue_d', 'purpose', 'addr_state']
    for dummy in dummy_lis:
        dum = loan[f'{dummy}']
        s.dum = 1
    #Transforming the features in the loan
    arr = loan[['annual_inc', 'delinq_amnt', 'tot_coll_amt', 'total_rec_late_fee']] + 1
    for i, x in enumerate(arr):
        arr[i] = math.log(x)
    loan.loc[['annual_inc', 'delinq_amnt', 'tot_coll_amt', 'total_rec_late_fee']] = arr
    loan.tax_liens = math.sqrt(loan.tax_liens)
    #Converting the interest string to a float
    loan.int_rate = float(loan.int_rate[:-1])
    #Turning the binary catagories into booleans
    if loan.term == ' 36 months':
        loan.term = True
    else:
        loan.term = False

    if loan.initial_list_status == 'w':
        loan.initial_list_status = True
    else:
        loan.initial_list_status = False
        
    if loan.application_type == 'Individual':
        loan.application_type = True
    else:
        loan.application_type = False
        
    if loan.disbursement_method == 'Cash':
        loan.disbursement_method = True
    else:
        loan.disbursement_method = False
    for index, value in loan.items():
        if index in s.columns:
            s[f'{index}'] = value
    return s

def evaluate_loan(loan, desired_interest):
    '''
    This functions pulls everything together. Returns True if the desired interest rate is met and the models find that the loan wont go into default, hardship, or settlement.
    '''
    if check_interest(loan, desired_interest):
        hardship_model = pickle.load( open( "models/final_hardship_model.pkl", "rb" ) )
        settle_model = pickle.load( open( "models/final_settle_model.pkl", "rb" ) )
        default_model = pickle.load( open( "models/final_default_model.pkl", "rb" ) )
        clean_loan = format_loan(loan)
        res_lis = [hardship_model.predict(clean_loan)[0], settle_model.predict(clean_loan)[0], default_model.predict(clean_loan)[0]]
        return not any (res_lis)
    else:
        return False

if __name__ == "__main__":
    #This is where you can add a stream of data or another data set. Using the first row of the first csv for demonstration.
    df = pd.read_csv('data/2016Q1.csv.gz')
    loan = df.iloc[0]
    desired_interest = 5 #Least interest rate acceptable
    if evaluate_loan(loan, desired_interest):
      print('Approved')
    else:
      print('Disapproved')
