class clf_reduction:
    '''
    Composing with a linear transformation to obtain outputs in passing from a space of one-hot encoded 
    features to a smaller such space.
    The probabilities for the new labels may be outputted as well if the original classifier has a predict_proba method, 
    or, alternatively, if a calibrated classifier is provided. 
    (In the multi-label setting, those probabilities should be interpreted as lower bounds).  
    '''
    def __init__(self, clf, T, clf_calibrated=None):
        self.clf=clf
        self.transformation=T
        if clf_calibrated==None:
            self.clf_calibrated=clf
        else:
            self.clf_calibrated=clf_calibrated
            
    def predict(self,X):
        import numpy as np
        return (np.matmul(self.clf.predict(X),self.transformation)>0).astype(int) 
    
    
    def predict_proba(self,X,multi_label=True):
        import numpy as np
        if multi_label:               #See https://stackoverflow.com/questions/41164305/numpy-dot-product-with-max-instead-of-sum
            return np.max(self.clf_calibrated.predict_proba(X)[:, :, None] * self.transformation[None, :, :], axis = 1)
        else:                         #The classification task is multi-class, not multi-label.
            return np.matmul(self.clf_calibrated.predict_proba(X),self.transformation)





