
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn import preprocessing
from scipy.special import expit
from models import pate_ganC
import numpy as np
import pandas as pd
import collections
import os


categorical = True
target_variable = "income"
train_data_path = "D:\\whole differential privacy\\data\\adult\\adult_processed_train.csv"
test_data_path  = "D:\\whole differential privacy\\data\\adult\\adult_processed_test.csv"
normalize_data  = True
downstream_task = "classification"
enable_privacy  = False
target_epsilon  = 8
target_delta    = 1e-5
save_synthetic  = False
output_data_path = ""

sigma = 2.
clip_coeff = 0.1
micro_batch_size = 8

num_epochs = 1000
batch_size = 64

model_name = "pate-gan"
clamp_lower = -0.01
clamp_upper = 0.01

lap_scale = 0.0001
num_teachers = 10
teacher_iters = 5
student_iters = 5
num_moments   = 100

# Loading the data
train = pd.read_csv(train_data_path)
test = pd.read_csv(test_data_path)

data_columns = [col for col in train.columns if col != target_variable]
if categorical:
    combined = train.append(test)
    config = {}
    for col in combined.columns:
        col_count = len(combined[col].unique())
        config[col] = col_count

class_ratios = None

if downstream_task == "classification":
    class_ratios = train[target_variable].sort_values().groupby(train[target_variable]).size().values/train.shape[0]

print(train.shape)

X_train = np.nan_to_num(train.drop([target_variable], axis=1).values)
y_train = np.nan_to_num(train[target_variable].values)
X_test = np.nan_to_num(test.drop([target_variable], axis=1).values)
y_test = np.nan_to_num(test[target_variable].values)

if normalize_data:
    X_train = expit(X_train)
    X_test = expit(X_test)

input_dim = X_train.shape[1]
z_dim = int(input_dim / 4 + 1) if input_dim % 4 == 0 else int(input_dim / 4)

conditional = (downstream_task == "classification")

# Training the generative model
if model_name == 'dp-wgan':
    Hyperparams = collections.namedtuple(
        'Hyperarams',
        'batch_size micro_batch_size clamp_lower clamp_upper clip_coeff sigma class_ratios lr num_epochs')
    Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None, None, None)

    model = dp_wganC.DP_WGAN(input_dim, z_dim, target_epsilon, target_delta, conditional)
    model.train(X_train, 
                y_train, 
                X_test, 
                y_test,
                Hyperparams(batch_size=batch_size, micro_batch_size=micro_batch_size,
                                              clamp_lower=clamp_lower, clamp_upper=clamp_upper,
                                              clip_coeff=clip_coeff, sigma=sigma, class_ratios=class_ratios, lr=
                                              5e-5, num_epochs=num_epochs), 
                private=enable_privacy)

if model_name == 'pate-gan':
    Hyperparams = collections.namedtuple(
        'Hyperarams',
        'batch_size num_teacher_iters num_student_iters num_moments lap_scale class_ratios lr')
    Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None)

    model = pate_ganC.PATE_GAN(input_dim, z_dim, num_teachers, target_epsilon, target_delta, conditional)
    accuracies = model.train(X_train, y_train, X_test, y_test, Hyperparams(batch_size=batch_size, num_teacher_iters=teacher_iters,
                                              num_student_iters=student_iters, num_moments=num_moments,
                                              lap_scale=lap_scale, class_ratios=class_ratios, lr=1e-4))

if model_name == 'imle' or model_name == 'dp-wgan' or model_name == 'pate-gan':
    syn_data = model.generate(X_train.shape[0], class_ratios)
    X_syn, y_syn = syn_data[:, :-1], syn_data[:, -1]


# # Testing the quality of synthetic data by training and testing the downstream learners

# # Creating downstream learners
# learners = []

# if opt.downstream_task == "classification":
#     names = ['LR', 'Random Forest', 'Neural Network', 'GaussianNB', 'GradientBoostingClassifier']

#     learners.append((LogisticRegression()))
#     learners.append((RandomForestClassifier()))
#     learners.append((MLPClassifier(early_stopping=True)))
#     learners.append((GaussianNB()))
#     learners.append((GradientBoostingClassifier()))

#     print("AUC scores of downstream classifiers on test data : ")
#     for i in range(0, len(learners)):
#         score = learners[i].fit(X_syn, y_syn)
#         pred_probs = learners[i].predict_proba(X_test)
#         auc_score = roc_auc_score(y_test, pred_probs[:, 1])
#         print('-' * 40)
#         print('{0}: {1}'.format(names[i], auc_score))

# else:
#     names = ['Ridge', 'Lasso', 'ElasticNet', 'Bagging', 'MLP']

#     learners.append((Ridge()))
#     learners.append((Lasso()))
#     learners.append((ElasticNet()))
#     learners.append((BaggingRegressor()))
#     learners.append((MLPRegressor()))

#     print("RMSE scores of downstream regressors on test data : ")
#     for i in range(0, len(learners)):
#         score = learners[i].fit(X_syn, y_syn)
#         pred_vals = learners[i].predict(X_test)
#         rmse = np.sqrt(mean_squared_error(y_test, pred_vals))
#         print('-' * 40)
#         print('{0}: {1}'.format(names[i], rmse))

# if opt.model != 'real-data':
#     if opt.save_synthetic:

#         if not os.path.isdir(opt.output_data_path):
#             raise Exception('Output directory does not exist')

#         X_syn_df = pd.DataFrame(data=X_syn, columns=data_columns)
#         y_syn_df = pd.DataFrame(data=y_syn, columns=[opt.target_variable])

#         syn_df = pd.concat([X_syn_df, y_syn_df], axis=1)
#         syn_df.to_csv(opt.output_data_path + "/synthetic_data.csv")
#         print("Saved synthetic data at : ", opt.output_data_path)



