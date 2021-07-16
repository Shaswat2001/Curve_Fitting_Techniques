import numpy as np
import math

class LMS:

    def __init__(self,model):
        self.solve_model=model 
    
    def select_best(self,consensus_set):

        med_list=[x for (x,y) in consensus_set]
        index_cs=med_list.index(min(med_list))
        parameters=consensus_set[index_cs][1]

        return parameters
    
    def solve_lms(self,X,Y):
        prob_best=0.95 
        sample_size=3
        no_iterations=3*math.log(1-0.95)/math.log(1-(0.99)**sample_size)
        current_itr=0     
        consensus=[]
        while current_itr<no_iterations:

            sample_idx=np.random.choice(np.arange(X.shape[0]),sample_size)
            X_rand=X[sample_idx]
            Y_rand=Y[sample_idx]
            parameters=self.solve_model(X_rand,Y_rand)
            _,median=self.make_prediction(parameters,X,Y)
            consensus.append((median,parameters))

            current_itr+=1
        
        best_param=self.select_best(consensus)

        return best_param

    def make_prediction(self,parameter,X,Y):

        Y_pred=np.sum(parameter*X,axis=1).reshape(-1,1)
        med=np.median((Y_pred-Y)**2)

        return Y_pred,med 