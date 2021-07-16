import numpy as np
import math 

class RANSAC:

    def __init__(self,model):
        self.solve_model=model 
    
    def select_best(self,consensus_set):

        length_lt=[len(x) for x in consensus_set]
        index_cs=length_lt.index(max(length_lt))
        best_data=consensus_set[index_cs]
        X_CS=np.array([x for (x,y) in best_data])
        Y_CS=np.array([y for (x,y) in best_data])
        parameters=self.solve_model(X_CS,Y_CS)

        return parameters

    def solve_ransac(self,X,Y,error=None):
        no_iterations=math.inf
        current_itr=0
        sample_size=3
        prob_best=0.95
        if error==None:
            threshold=3*np.std(Y)/5
        else:
            threshold=3*np.std(error)/5
        
        consensus=[]
        while current_itr<no_iterations:

            sample_idx=np.random.choice(np.arange(X.shape[0]),sample_size)
            test_idx=np.setdiff1d(np.arange(X.shape[0]),sample_idx)
            X_rand=X[sample_idx]
            Y_rand=Y[sample_idx]
            X_test=X[test_idx]
            Y_test=Y[test_idx]
            parameters=self.solve_model(X_rand,Y_rand)
            CS=[]
            for i in range(X_test.shape[0]):
                Y_pred=np.sum(parameters*X_test[i])
                if np.abs(Y_pred-Y_test[i])<threshold:
                    CS.append((X_test[i],Y_test[i]))
            consensus.append(CS)

            prob_inlier=len(CS)/X.shape[0]
            current_itr+=1
            no_iterations=math.log(1-prob_best)/math.log(1-prob_inlier**sample_size)
        
        best_param=self.select_best(consensus)

        return best_param

    def make_prediction(self,parameter,X,Y):

        Y_pred=np.sum(parameter*X,axis=1).reshape(-1,1)
        error=np.abs(Y_pred-Y)

        return Y_pred,error 