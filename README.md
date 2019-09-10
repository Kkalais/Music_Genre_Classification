# Music_Genre_Classification
Machine Learning Algorithms for Music Genre Classification

## Dataset
Experiments were carried out on a dataset for music analysis called FMA (Free Music Archive) by Swiss Data Science Center: [FMA_dataset](https://github.com/mdeff/fma).

## Prerequisites
Install the libraries below used by the project by entering in console the following command:

  ```pip3 install pandas matplotlib keras scikit-learn numpy more-tertools seaborn xgboost```
  
Clone the repository locally by entering in console the following command:

  ```git clone https://github.com/Kkalais/Music_Genre_Classification.git```
 
 ## Run
 
We are using Gradient Boosting, **Support-Vectors Machine, Random Forest, and Multilayer Perceptron Neural Network** to classify the music samples into 16 different music genres.
 
In order to run the code using the above-mentioned algorithms just enter in console the following commands :
 
  ```python3 main.py gradient_goosting```
 
  ```python3 main.py svm```
 
  ```python3 main.py random_forest```
 
  ```python3 main.py k_neighbors```
  
  ```python3 main.py mlp```
  
  ```python3 main.py logistic_regression```
  
  ```python3 main.py xgboost```
  
  
respectively.

There is also a mode that runs all four algorithms consecutively, and produces a bar plot to compare the algorithms' results. Please enter in console:

```python3 main.py comparative```

## Authors

* **Konstantinos Kalais**, *Developer* 
