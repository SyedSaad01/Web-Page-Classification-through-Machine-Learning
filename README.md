Problem Statement:
    Identify whether a web page is malicious or benign using the input features and 
    classify them accordingly.

Size of Dataset: i    
    650 MBs

Number of features:
    11(input)+1(output) = 12 features

Pre-processing:
    1) Encoding of categorical features
    2) Scaling
    3) Handling Outliers
    4) Handling Imbalance Dataset
    5) Dimentionality Reduction
    6) Handling missing values

Algorithm Applied:
    Random Forest Classifier.
    Results:
        Model gives 99.80% accuracy. Precision & recall for both the classes came out to be 1, with perfect area of 1 under the ROC curve.


ML Model Integration with Django:
    The above model was integrated with python's django framework to design end-to-end web application.                    