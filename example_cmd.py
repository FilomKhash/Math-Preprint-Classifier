print('Importing necessary libraries...\n')
import end_to_end
from end_to_end import math_classifier

#Importing classes that we had defined ourselves.
from Concat import ConcatModels
from Reduction import clf_reduction


print('Loading models...\n')
clf=math_classifier()
    
print('Scraping random articles:')
clf.predict()      

