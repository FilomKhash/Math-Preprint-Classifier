class ConcatModels:
    '''
    Concatenate provided binary classifiers which have a predict_proba method.
    '''
    def __init__(self,list_models):
        self.models=list_models
        
    def predict_proba(self,X):
        import numpy as np
        return np.transpose(np.asarray([model.predict_proba(X)[:,1] for model in self.models]))




