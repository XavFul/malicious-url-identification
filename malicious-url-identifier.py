import numpy as np
import pandas as pd
from urllib.parse import urlparse
import ipaddress
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import validation_curve, learning_curve, GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

#
# Feature extraction definitions
#

# Check for IP address in the URL
def check_IP(url):
  try:
    domain = urlparse(url).netloc.split(":")[0]
    ipaddress.ip_address(domain)
    ip = 1
  except ValueError:
    ip = 0
  return ip

# Check for http in the start of the URL
def check_HTTP(url):
  if url.startswith('http:\\'):
    http = 1
  else:
    http = 0
  return http

# Check for https in the start of the URL
def check_HTTPS(url):
  if url.startswith('https:\\'):
    https = 1
  else:
    https = 0
  return https

# Check for html in the URL
def check_HTML(url):
  url_string = url
  if "html" in url_string.lower():
    html = 1
  else:
    html = 0
  return html

# Check for php in the URL
def check_PHP(url):
  url_string = url
  if "php" in url_string.lower():
    php = 1
  else:
    php = 0
  return php

# Check for @ in the URL
def check_atsign(url):
  url_string = url
  if "@" in url_string.lower():
    atsign = 1
  else:
    atsign = 0
  return atsign

# Get the length of the URL
def count_length(url):
    return len(url)

# Get the length of the domain of the URL
def count_domain_length(url):
    domain = urlparse(url).netloc
    return len(domain)

# Count the number of slashes in the path of the URL
def count_path_slash(url):
    path = urlparse(url).path
    return path.count('/')

# Get the number of '.'s in the domain of the URL
def count_domain_dots(url):
    domain = urlparse(url).netloc
    return domain.count('.')

# Get the number of '-'s in the URL
def count_hyphens(url):
    return url.count('-')

# Get the number of non-alphanumeric characters in the URL
def count_non_alpha(url):
    non_alpha_count = 0
    for char in url:
        if not char.isalnum():
            non_alpha_count += 1
    return non_alpha_count

#
# Feature extraction
#

df['IP'] = df['url'].apply(check_IP)
df['HTTP'] = df['url'].apply(check_HTTP)
df['HTTPS'] = df['url'].apply(check_HTTPS)
df['HTML'] = df['url'].apply(check_HTML)
df['PHP'] = df['url'].apply(check_PHP)
df['AtSign'] = df['url'].apply(check_atsign)
df['Length'] = df['url'].apply(count_length)
df['DomainLength'] = df['url'].apply(count_domain_length)
df['PathSlashes'] = df['url'].apply(count_path_slash)
df['DomainDots'] = df['url'].apply(count_domain_dots)
df['Hyphens'] = df['url'].apply(count_hyphens)
df['NonAlphaChars'] = df['url'].apply(count_non_alpha)

# Remove URL column as we performed feature extraction and no longer need them
df.drop(columns=['url'], inplace=True)

# Export processed data to new CSV
df.to_csv('features_data_set.csv', index=False)

# Prepare data for model training
X = df.drop(columns=['type'])
Y = df['type']

# Encode the target variable
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)


X_rand, _, Y_rand, _ = train_test_split(X, Y, test_size=0.9, random_state=42)

# XGBoost model training and evaluation
def xgb_train_and_evaluate(X_train, Y_train, X_test, Y_test, model_name):
    start_time = time.time()
    
    model = XGBClassifier(learning_rate=0.2, max_depth=7, n_estimators=200)
    model.fit(X_train, Y_train)

    train_time = time.time() - start_time
    print(f"{model_name} Training Time: {train_time} seconds")

    start_time = time.time()
    Y_pred = model.predict(X_test)
    test_time = time.time() - start_time
    
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"{model_name} Model Accuracy: {accuracy}")
    print(f"{model_name} Prediction Time: {test_time} seconds")

    # Print precision, recall, and f1 scores
    precision = precision_score(Y_test, Y_pred, average='weighted')
    recall = recall_score(Y_test, Y_pred, average='weighted')
    f1 = f1_score(Y_test, Y_pred, average='weighted')

    print(f"{model_name} Precision: {precision}")
    print(f"{model_name} Recall: {recall}")
    print(f"{model_name} F1 Score: {f1}")

    return model

# XGBoost model training and evaluation
X_rand_train, X_rand_test, Y_rand_train, Y_rand_test = train_test_split(X_rand, Y_rand, test_size=0.2, random_state=42)
xgb_model = xgb_train_and_evaluate(X_rand_train, Y_rand_train, X_rand_test, Y_rand_test, 'XGB')

# LightGBM model training and evaluation
def lgbm_train_and_evaluate(X_train, Y_train, X_test, Y_test, model_name):
    start_time = time.time()
    
    model = LGBMClassifier(learning_rate=0.1, max_depth=15, n_estimators=100, num_leaves=128, verbosity=-1)
    model.fit(X_train, Y_train)

    train_time = time.time() - start_time
    print(f"{model_name} Training Time: {train_time} seconds")

    start_time = time.time()
    Y_pred = model.predict(X_test)
    test_time = time.time() - start_time
    
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"{model_name} Model Accuracy: {accuracy}")
    print(f"{model_name} Prediction Time: {test_time} seconds")

    # Print precision, recall, and f1 scores
    precision = precision_score(Y_test, Y_pred, average='weighted')
    recall = recall_score(Y_test, Y_pred, average='weighted')
    f1 = f1_score(Y_test, Y_pred, average='weighted')

    print(f"{model_name} Precision: {precision}")
    print(f"{model_name} Recall: {recall}")
    print(f"{model_name} F1 Score: {f1}")
    
    return model

lgbm_model = lgbm_train_and_evaluate(X_rand_train, Y_rand_train, X_rand_test, Y_rand_test, 'LGBM')

# Define the hyperparameter grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200],
}

# Instantiate the classifier
xgb = XGBClassifier()

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy', verbose=0)

# Perform grid search
grid_search.fit(X_rand_train, Y_rand_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

param_grid = {
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [7, 10, 15],
    'n_estimators': [100, 200, 300],
    'num_leaves': [31, 64, 128]
}

grid_search = GridSearchCV(lgb.LGBMClassifier(), param_grid, cv=5, verbose=0)
grid_search.fit(X_train_lc, Y_train_lc)

best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Confusion Matrix and Metric Plots
class_names = ['benign', 'phishing', 'malware', 'defacement']

def plot_confusion_matrix(Y_test, Y_pred, class_names, model_name):
    cm = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(8, 6))

    # Logarithmic normalization for better visibility of lower values
    log_norm = np.log1p(cm)

    # Define a colormap
    cmap = sns.light_palette("navy", as_cmap=True)

    plt.imshow(log_norm, interpolation='nearest', cmap=cmap)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center",
                     verticalalignment="center", color="white" if cm[i, j] > np.max(cm) / 2 else "black")

    plt.show()

# Define metrics variable
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Function to plot metrics
def plot_metrics(metrics, values1, values2, model_name1, model_name2):
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(metrics))

    plt.bar(index, values1, width=bar_width, label=model_name1)
    plt.bar(index + bar_width, values2, width=bar_width, label=model_name2)

    plt.title('Model Comparison - Evaluation Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.ylim(0.923, 0.93)
    plt.xticks(index + bar_width / 2, metrics)
    plt.legend()
    plt.show()

# Evaluate and plot metrics for LightGBM and XGBoost models

# Evaluate and plot metrics for LightGBM model
Y_pred_lgbm = lgbm_model.predict(X_rand_test)
plot_confusion_matrix(Y_rand_test, Y_pred_lgbm, class_names, 'LGBM')

accuracy_lgbm = accuracy_score(Y_rand_test, Y_pred_lgbm)
precision_lgbm = precision_score(Y_rand_test, Y_pred_lgbm, average='weighted')
recall_lgbm = recall_score(Y_rand_test, Y_pred_lgbm, average='weighted')
f1_lgbm = f1_score(Y_rand_test, Y_pred_lgbm, average='weighted')

values_lgbm = [accuracy_lgbm, precision_lgbm, recall_lgbm, f1_lgbm]

# Evaluate and plot metrics for XGBoost model
Y_pred_xgb = xgb_model.predict(X_rand_test)
plot_confusion_matrix(Y_rand_test, Y_pred_xgb, class_names, 'XGB')

accuracy_xgb = accuracy_score(Y_rand_test, Y_pred_xgb)
precision_xgb = precision_score(Y_rand_test, Y_pred_xgb, average='weighted')
recall_xgb = recall_score(Y_rand_test, Y_pred_xgb, average='weighted')
f1_xgb = f1_score(Y_rand_test, Y_pred_xgb, average='weighted')

values_xgb = [accuracy_xgb, precision_xgb, recall_xgb, f1_xgb]

# Plot metrics for both models on the same graph
plot_metrics(metrics, values_lgbm, values_xgb, 'LGBM', 'XGB')

# Define the parameter values to be tested
param_range = ['IP', 'HTTP', 'HTTPS', 'HTML', 'PHP', 'AtSign', 'Length', 'DomainLength', 'PathSlashes', 'DomainDots', 'Hyphens', 'NonAlphaChars']

# Subsample 10% of the dataset for learning curve analysis with random instance selection
X_rand, _, Y_rand, _ = train_test_split(X, Y, test_size=0.9, random_state=42)

# Use the specified hyperparameters to train the final model
best_xgb = XGBClassifier(learning_rate=0.2, max_depth=7, n_estimators=200)
best_xgb.fit(X_rand, Y_rand)

# Calculate training and test scores across different parameter values
train_scores, test_scores = validation_curve(
    best_xgb,
    X, Y,  # Your data
    param_name="learning_rate",  # The parameter you want to test
    param_range=param_range,
    cv=5  # Number of cross-validation folds
)

# Calculate the mean and standard deviation of training and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Calculate learning curves using only 10% of the dataset
train_sizes, train_scores_lc, test_scores_lc = learning_curve(
    XGBClassifier(),  # Your classifier
    X_rand, Y_rand,  # Subsampled training data
    cv=5,  # Number of cross-validation folds
    train_sizes=np.linspace(0.1, 1.0, 10),  # Training set sizes to be tested
)

# Calculate the mean and standard deviation of training and test scores
train_scores_lc_mean = np.mean(train_scores_lc, axis=1)
train_scores_lc_std = np.std(train_scores_lc, axis=1)
test_scores_lc_mean = np.mean(test_scores_lc, axis=1)
test_scores_lc_std = np.std(test_scores_lc, axis=1)

# Plot the curves on the same graph & standardize size
plt.figure(figsize=(12, 8))
plt.title("Validation and Learning Curves")
plt.xlabel("Parameter Values / Training Examples")
plt.ylabel("Score")
plt.ylim(0.875, 1.0)
lw = 2

# Plot curves for XGB
plt.plot(train_sizes, train_scores_lc_mean, label="XGB Training score", color="darkred", lw=lw)
plt.fill_between(train_sizes, train_scores_lc_mean - train_scores_lc_std, train_scores_lc_mean + train_scores_lc_std, alpha=0.2, color="darkred", lw=lw)

plt.plot(train_sizes, test_scores_lc_mean, label="XGB Cross-validation score", color="darkblue", lw=lw)
plt.fill_between(train_sizes, test_scores_lc_mean - test_scores_lc_std, test_scores_lc_mean + test_scores_lc_std, alpha=0.2, color="darkblue", lw=lw)

plt.legend(loc="best")
plt.show()

# Plot the curves on the same graph & standardize size
plt.figure(figsize=(12, 8))
plt.title("Validation and Learning Curves")
plt.xlabel("Parameter Values / Training Examples")
plt.ylabel("Score")
plt.ylim(0.875, 1.0)
lw = 2

# Use the specified hyperparameters to train the final LightGBM model
best_lgbm = LGBMClassifier(learning_rate=0.1, max_depth=15, n_estimators=100, num_leaves=128, verbose=-1)
best_lgbm.fit(X_rand, Y_rand)

# Calculate training and test scores across different parameter values for LightGBM
train_scores_lgbm, test_scores_lgbm = validation_curve(
    best_lgbm,
    X, Y,
    param_name="learning_rate",
    param_range=param_range,
    cv=5
)

# Calculate the mean and standard deviation of training and test scores for LightGBM
train_scores_mean_lgbm = np.mean(train_scores_lgbm, axis=1)
train_scores_std_lgbm = np.std(train_scores_lgbm, axis=1)
test_scores_mean_lgbm = np.mean(test_scores_lgbm, axis=1)
test_scores_std_lgbm = np.std(test_scores_lgbm, axis=1)

# Calculate learning curves using only 10% of the dataset for LightGBM
train_sizes_lgbm, train_scores_lc_lgbm, test_scores_lc_lgbm = learning_curve(
    LGBMClassifier(learning_rate=0.1, max_depth=15, n_estimators=100, num_leaves=128, verbose=-1),
    X_rand, Y_rand,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
)

# Calculate the mean and standard deviation of training and test scores for LightGBM
train_scores_lc_mean_lgbm = np.mean(train_scores_lc_lgbm, axis=1)
train_scores_lc_std_lgbm = np.std(train_scores_lc_lgbm, axis=1)
test_scores_lc_mean_lgbm = np.mean(test_scores_lc_lgbm, axis=1)
test_scores_lc_std_lgbm = np.std(test_scores_lc_lgbm, axis=1)

# Plot curves for LGBM
plt.plot(train_sizes_lgbm, train_scores_lc_mean_lgbm, label="LightGBM Training score", color="green", lw=lw)
plt.fill_between(train_sizes_lgbm, train_scores_lc_mean_lgbm - train_scores_lc_std_lgbm, train_scores_lc_mean_lgbm + train_scores_lc_std_lgbm, alpha=0.2, color="green", lw=lw)

plt.plot(train_sizes_lgbm, test_scores_lc_mean_lgbm, label="LightGBM Cross-validation score", color="orange", lw=lw)
plt.fill_between(train_sizes_lgbm, test_scores_lc_mean_lgbm - test_scores_lc_std_lgbm, test_scores_lc_mean_lgbm + test_scores_lc_std_lgbm, alpha=0.2, color="orange", lw=lw)

plt.legend(loc="best")
plt.show()
