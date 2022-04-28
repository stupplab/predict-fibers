
## Deep Learning model to classify a Peptide Amphiphile assembly into Fiber and Nonfiber given its peptide sequence

Launch the fiber predicting Jupyter notebook-main.ipynb-by 
- click -> 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/stupplab/predict-fibers/HEAD) 
- Then double-click `main.ipynb` shown on the left side

\
To predict, simply add the peptide sequences in `seqs` variable at the bottom of the code, then click the run button. The result generated is the fiber-nonfiber prediction along with the probability score of the sequence being a fiber according to the model.\
Note that all the peptide sequences are assumed to be prepended by the C16 alkyl tail with unit charge per molecule.

\
The model is trained using `train.csv` and evaluated on `test.csv`, both containing equal mix of Fibers and Nonfibers. 
Details are as below
- Train set: 4619 Fibers | 4038 Nonfibers
- Test set: 1180 Fibers | 985 Nonfibers
    - True Positive  1087
    - False Positive   83
    - True Negative   902
    - False Negative   93
    - Precision       93%
    - Accuracy        92%
