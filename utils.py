import numpy as np
import matplotlib.pyplot as plt
import uuid



def generate_run_id():
    return str(uuid.uuid4())


#getting class balance
def get_class_balance(y):
    if hasattr(y,'values'):
        y=y.values
    unique,counts=np.unique(y,return_counts=True)
    total=len(y)
    balance={}
    for label,count in zip(unique,counts):
        balance[int(label)]={'count':int(count),'percentage':float(count)/total*100}

    return balance
