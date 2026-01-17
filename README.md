# Iris Species Classification: An Algorithmic Comparison 🌸

##### 

###### This project explores the classic Iris dataset by comparing three distinct machine learning approaches: \*\*Logistic Regression\*\*, \*\*K-Nearest Neighbors (KNN)\*\*, and \*\*Gaussian Naive Bayes\*\*. The goal was to evaluate model robustness when training data is intentionally limited.

###### 

### 🚀 Project Overview

###### Most tutorials use an 80/20 train-test split on the Iris dataset, often resulting in a "perfect" 100% accuracy. To make the challenge more realistic and observe how different algorithms behave under stress, I utilized a \*\*30% training and 70% testing split\*\*.

##### 

### Algorithms Compared:

###### 1\. \*\*Logistic Regression:\*\* A linear approach that attempts to find the best decision boundaries between species.

###### 2\. \*\*K-Nearest Neighbors (KNN):\*\* A distance-based algorithm that classifies points based on local similarity.

###### 3\. \*\*Gaussian Naive Bayes:\*\* A probabilistic model that assumes feature independence and uses Gaussian distributions.

##### 

### 🛠️ Key Features \& Methodology

###### \* \*\*Feature Scaling:\*\* Implemented `StandardScaler` to ensure distance-based models (KNN) aren't biased by feature ranges.

###### \* \*\*Pipelines:\*\* Used Scikit-Learn `Pipeline` to wrap scaling and model training into a single, clean workflow.

###### \* \*\*Hyperparameter Tuning:\*\* Conducted a `GridSearchCV` to find the optimal `n\_neighbors` for KNN.

###### \* \*\*Multiclass Metrics:\*\* Solved the `ValueError: Target is multiclass but average='binary'` by implementing `macro` and `weighted` averaging for Precision and Recall.

###### 

### 📊 Results \& Insights

###### | Algorithm | Key Strength on this Dataset |

###### | KNN | Top Performer The physical clustering of Iris features makes "local similarity" a highly reliable predictor. |

###### | Logistic Regression | Performs well on Setosa but shows slight overlap errors between Versicolor and Virginica. |

###### | Naive Bayes | Extremely fast, though slightly less accurate as it ignores correlations between petal and sepal dimensions. |

###### 

### The "Aha!" Moment: Why KNN Won

###### Even with only 45 samples (30% training), KNN remained robust. In the Iris dataset, flowers of the same species are physically close in the feature space. While Logistic Regression struggled to find the perfect global line with limited data, KNN succeeded by simply looking at the "nearest neighbors."

##### 

### 📈 Visualizations

###### Feature Scaling Effect

###### Scaling was critical for KNN. Without it, \*\*Petal Length\*\* (range 1-7) would have carried 17x more weight than \*\*Sepal Width\*\* (range 2-4.4).

###### 

##### 

### Confusion Matrix

###### The following matrix shows how the tuned KNN model successfully distinguished between the species:

###### !\[Confusion Matrix](figure.png)

##### 

### 💻 How to Run

###### 1\. Clone the repository: `git clone [https://github.com/Kafkaesque-08/iris-dataset-with-knn-logistic-naive.git`](https://github.com/kafkaesque-08/Projects.git)


##### 

### 🧠 Skills Demonstrated

###### \* Data Preprocessing \& Cleaning

###### \* Feature Engineering (Standardization)

###### \* Model Selection \& Comparison

###### \* Hyperparameter Optimization (Grid Search)

###### \* Advanced Model Evaluation (Multiclass Metrics)



