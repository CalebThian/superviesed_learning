# How to launch
As PyInstaller has some problem unsolved, the exe file cannot run
Please create a virtual environment and install the requirements package, then run the code, thank you!
Before run the code, please make sure train.csv under the same directory with all the \*.py
Notice that myRandomForest Classifier need more amount of time.

conda create env
conda activate env
pip install -r requirements.txt
python main.py

Sample output:

Naive Bayes Clasifier:
Recall = 99.74%
Precision = 6.66%
Accuracy = 7.22%
F1-score = 12.49%
Naive Bayes Clasifier: F1-score
->3-fold cross validation:12.13%
->5-fold cross validation:12.13%
->10-fold cross validation:12.13%
my Random Forest Classifier:
Recall = 0.00%
Precision = 0.00%
Accuracy = 93.54%
F1-score = 0.00%
my Random Forest Classifier: F1-score
->3-fold cross validation:0.00%
->5-fold cross validation:0.00%
->10-fold cross validation:0.00%
Random Forest Classifier of scikit-learn:
Recall = 0.00%
Precision = 0.00%
Accuracy = 93.33%
F1-score = 0.00%
Random Forest Classifier of scikit-learn: F1-score
->3-fold cross validation:0.13%
->5-fold cross validation:0.20%
->10-fold cross validation:0.07%
XGBoost Classifier:
Recall = 0.13%
Precision = 33.33%
Accuracy = 93.35%
F1-score = 0.26%
XGBoost Classifier: F1-score
->3-fold cross validation:0.00%
->5-fold cross validation:0.27%
->10-fold cross validation:0.20%
CatBoost Classifier:
Recall = 0.00%
Precision = 0.00%
Accuracy = 93.53%
F1-score = 0.00%
CatBoost Classifier: F1-score
->3-fold cross validation:0.13%
->5-fold cross validation:0.13%
->10-fold cross validation:0.13%
Light Boost Classifier:
Recall = 58.14%
Precision = 8.90%
Accuracy = 60.27%
F1-score = 15.44%
Light Boost Classifier: F1-score
->3-fold cross validation:16.36%
->5-fold cross validation:16.05%
->10-fold cross validation:16.16%

To get visualize output, please run main.ipynb