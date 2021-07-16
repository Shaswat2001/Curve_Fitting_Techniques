import math 
import numpy as np 

class Least_Sqaure:

    def __init__(self,model):
        self.solve_model=model 
    
    def solve_least_sqaure(self,X,Y):

        parameters=self.solve_model(X,Y)
        return parameters
    
    def make_prediction(self,parameter,X,Y):

        Y_pred=np.sum(parameter*X,axis=1).reshape(-1,1)
        med=np.median(np.abs(Y_pred-Y)**2)

        return Y_pred,med 