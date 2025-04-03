# Scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn import tree
import diffprivlib.models as dp
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from art.estimators.classification.pytorch import PyTorchClassifier

from sklearn.tree import DecisionTreeClassifier

def log(y, pre):
    e = 0.0000001
    pre = np.clip(pre, e, 1 - e)
    return - y * np.log(pre) - (1 - y) * np.log(1 - pre)

class BaseModel(object):

    def __init__(self):
        pass

    def bare_model(self):
        pass

    def train(self):
        pass

class DPLRModel(BaseModel):
    model_name = 'Differentially Private Logistic Regression'
    
    def train (self, dataset_train, SCALER, ATTACK):
        print('[INFO]: training dp logistic regression')
        lower_bounds = np.percentile(dataset_train.features, 1, axis=0)
        upper_bounds = np.percentile(dataset_train.features, 99, axis=0)
        if ATTACK == "mia1":
            if SCALER:
                model = make_pipeline(StandardScaler(), dp.LogisticRegression(solver='liblinear', random_state=1, epsilon=1, bounds=(lower_bounds,upper_bounds)))
            else:
                model = make_pipeline(dp.LogisticRegression(solver='liblinear', random_state=1))
            
            fit_params = {'logisticregression_sample_weight': dataset_train.instance_weights}
            model.fit(dataset_train.features, dataset_train.labels.ravel(), **fit_params)
        elif ATTACK == "mia2":
            model = dp.LogisticRegression(solver='liblinear', random_state=1, epsilon=10, bounds=(lower_bounds,upper_bounds))
            model.fit(dataset_train.features, dataset_train.labels.ravel(), sample_weight=dataset_train.instance_weights)
        return model

class LRModel(BaseModel):

    model_name = 'Logistic Regression'

    def train (self, dataset_train, SCALER, ATTACK):
        print ('[INFO]: training logistic regression')
        if ATTACK == "mia1":
            if SCALER:
                model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
            else:
                model = make_pipeline(LogisticRegression(solver='liblinear', random_state=1))
            fit_params = {'logisticregression__sample_weight': dataset_train.instance_weights}
            model.fit(dataset_train.features, dataset_train.labels.ravel(), **fit_params)
        elif ATTACK == "mia2":
            model = LogisticRegression(solver='liblinear', random_state=1)
            model.fit(dataset_train.features, dataset_train.labels.ravel(), sample_weight=dataset_train.instance_weights)

        return model

    def bare_model(self):

        model = LogisticRegression(solver='liblinear', random_state=1)

        return model

def weighted_resample(X, y, weights):
    """
    Resample X and y with replacement according to the provided weights.
    """
    weights = weights / np.sum(weights)
    n_samples = len(y)
    indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True, p=weights)
    return X[indices], y[indices]

class MLPClassifierWithWeightWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(128, 64, 32), activation='relu',
                 solver='adam', alpha=1e-4, learning_rate='adaptive',
                 max_iter=500, random_state=42, early_stopping=True):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.model_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=self.early_stopping
        )

    def fit(self, X, y, sample_weight=None):
        # If sample_weight is provided, use weighted resampling.
        if sample_weight is not None:
            X, y = weighted_resample(np.array(X), np.array(y), sample_weight)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
    
class MLP(BaseModel):
    model_name = "MLP Classifier"
    
    def train (self, dataset_train, sample_weights, ATTACK):
        print('[INFO]: training differentially private random forest')
        model = MLPClassifierWithWeightWrapper()
        model.fit(dataset_train.features, dataset_train.labels.ravel(), sample_weight=sample_weights)
        return model
    
class DPRF(BaseModel):
    model_name = 'Differentially Private Random Forest'
    
    def train (self, DATASET, dataset_train, SCALER, ATTACK):
        print ('[INFO]: training differentially private random forest')
        if ATTACK == "mia1":
            if SCALER:
                model = make_pipeline(StandardScaler(), dp.RandomForestClassifier(random_state=1, epsilon=1, bounds=(0,1)))
            else:
                model = make_pipeline(dp.RandomForestClassifier(random_state=1, epsilon=1, bounds=(0,1)))
            
            fit_params = {'randomforestclassifier__sample_weight': dataset_train.instance_weights}
            model.fit(dataset_train.features, dataset_train.labels.ravel(), **fit_params)
        elif ATTACK == "mia2":
            lower_bounds = 0
            upper_bounds = 1
            
            if DATASET == "bank" or DATASET.startswith("german") or DATASET == "meps19":
                classifier = dp.RandomForestClassifier(random_state=1,  epsilon=1, bounds=(lower_bounds, upper_bounds), max_depth=15)
            elif DATASET.startswith("compas"):
                classifier = dp.RandomForestClassifier(random_state=1,  epsilon=1, bounds=(lower_bounds, upper_bounds), max_depth=6)
            elif DATASET == "law_sex" or DATASET == "law_race":
                classifier = dp.RandomForestClassifier(random_state=1,  epsilon=1, bounds=(lower_bounds, upper_bounds), max_depth=7)
            elif DATASET.startswith("law"):
                classifier = dp.RandomForestClassifier(random_state=1,  epsilon=1, bounds=(lower_bounds, upper_bounds), max_depth=3)
            classifier.fit(dataset_train.features, dataset_train.labels.ravel(), sample_weight=dataset_train.instance_weights)
        return classifier

class RFModel(BaseModel):

    model_name = 'Random Forest'
    def __init__(self, n_est = 1000, min_leaf=5):
        self.n_est = n_est
        self.min_leaf = min_leaf

    def train (self, DATASET, dataset_train, SCALER, ATTACK):
        print ('[INFO]: training random forest')
        if ATTACK == "mia1":
            if SCALER:
                model = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=1))
            else:
                model = make_pipeline(RandomForestClassifier(random_state=1))
            
            fit_params = {'randomforestclassifier__sample_weight': dataset_train.instance_weights}
            model.fit(dataset_train.features, dataset_train.labels.ravel(), **fit_params)
        elif ATTACK == "mia2":
            if DATASET == "bank" or DATASET.startswith("german") or DATASET == "meps19":
                classifier = RandomForestClassifier(random_state=1, max_depth=15)
            elif DATASET.startswith("compas"):
                classifier = RandomForestClassifier(random_state=1, max_depth=6)
            elif DATASET == "law_sex" or DATASET == "law_race":
                classifier = RandomForestClassifier(random_state=1, max_depth=7)
            elif DATASET.startswith("law"):
                classifier = RandomForestClassifier(random_state=1, max_depth=3)
            classifier.fit(dataset_train.features, dataset_train.labels.ravel(), sample_weight=dataset_train.instance_weights)
        return classifier

    def bare_model(self):

        model = RandomForestClassifier(n_estimators=self.n_est, min_samples_leaf=self.min_leaf)

        return model

    
class DTModel(BaseModel):

    model_name = 'Decision Tree'
    def __init__(self, depth = 10):
        self.depth = depth

    def train (self, dataset_train, SCALER, ATTACK):
        print ('[INFO]: training decision tree')
        if ATTACK == "mia1":
            if SCALER:
                model = make_pipeline(StandardScaler(), tree.DecisionTreeClassifier(max_depth=self.depth))
            else:
                model = make_pipeline(tree.DecisionTreeClassifier(max_depth=self.depth))
            
            fit_params = {'decisiontreeclassifier__sample_weight': dataset_train.instance_weights}
            model.fit(dataset_train.features, dataset_train.labels.ravel(), **fit_params)
        elif ATTACK == "mia2":
            model = DecisionTreeClassifier(min_samples_leaf=10, max_depth=10)
            model.fit(dataset_train.features, dataset_train.labels.ravel(), sample_weight=dataset_train.instance_weights)
        return model

    def bare_model(self):

        model = tree.DecisionTreeClassifier(max_depth=self.depth)

        return model


class SVMModel(BaseModel):

    model_name = 'SVM'

    def train (self, dataset_train, SCALER):
        print ('[INFO]: training svm')
        if SCALER:
            model = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
        else:
            model = make_pipeline(SVC(gamma='auto', probability=True))
        fit_params = {'svc__sample_weight': dataset_train.instance_weights}
        model.fit(dataset_train.features, dataset_train.labels.ravel(), **fit_params)

        return model

    def bare_model(self):

        model = SVC(gamma='auto',probability=True)

        return model


# class NNModel(BaseModel):

#     model_name = 'Neural Network'

#     def train_old (self, dataset_train, SCALER):
#         print ('[INFO]: training neural network')
#         if SCALER:
#             model = make_pipeline(scaler, MLPClassifier(solver='adam', hidden_layer_sizes=(32,16,8), random_state=1))
#         else:
#             model = make_pipeline(MLPClassifier(solver='adam', hidden_layer_sizes=(32,16,8), random_state=1))
#             # model = MLPClassifier(solver='adam', alpha=0.001, hidden_layer_sizes=(32,16,8,2), random_state=1)

#         fit_params = {'mlpclassifier__sample_weight': dataset_train.instance_weights}
#         model.fit(dataset_train.features, dataset_train.labels.ravel())

#         return model

#     # train nn model using torch
#     def train(self, dataset_train, SCALER):
        
#         class ModelToAttack(nn.Module):

#             def __init__(self, num_classes, num_features):
#                 super(ModelToAttack, self).__init__()

#                 self.fc1 = nn.Sequential(
#                         nn.Linear(num_features, 512),
#                         nn.Tanh(), )

#                 self.fc2 = nn.Sequential(
#                         nn.Linear(512, 256),
#                         nn.Tanh(), )

#                 self.fc3 = nn.Sequential(
#                     nn.Linear(256, 128),
#                     nn.Tanh(),
#                 )

#                 self.classifier = nn.Linear(128, num_classes)

#             def forward(self, x):
#                 out = self.fc1(x)
#                 out = self.fc2(out)
#                 out = self.fc3(out)
#                 return self.classifier(out)

#         device = (
#             "cuda"
#             if torch.cuda.is_available()
#             else "cpu"
#         )
#         # print(f"Using {device} device")
#         mlp_model = ModelToAttack(2, dataset_train.features.shape[1]).to(device)
#         #mlp_model = torch.nn.DataParallel(mlp_model)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)

#         class FDataset(Dataset):
#             def __init__(self, x, y=None):
#                 self.x = torch.from_numpy(x.astype(np.float64)).type(torch.FloatTensor)

#                 if y is not None:
#                     self.y = torch.from_numpy(y.astype(np.int8)).type(torch.LongTensor)
#                 else:
#                     self.y = torch.zeros(x.shape[0])

#             def __len__(self):
#                 return len(self.x)

#             def __getitem__(self, idx):
#                 if idx >= len(self.x):
#                     raise IndexError("Invalid Index")

#                 return self.x[idx], self.y[idx]

#         train_set = FDataset(dataset_train.features, dataset_train.labels.ravel())
#         train_loader = DataLoader(train_set, batch_size=800, shuffle=True, num_workers=0)

#         # tensor_x = torch.Tensor(dataset_train.features) # transform to torch tensor
#         # tensor_y = torch.Tensor(dataset_train.labels.ravel())

#         # my_dataset = TensorDataset(tensor_x,tensor_y.type(torch.LongTensor)) # create your datset
#         # train_loader = DataLoader(my_dataset) # create your dataloader

#         mlp_model.train()
#         for epoch in range(10):
#             for (x, targets) in train_loader:
#                 # input, targets = torch.autograd.Variable(x), torch.autograd.Variable(targets)
#                 x, targets = x.to(device), targets.to(device)
#                 optimizer.zero_grad()
#                 outputs = mlp_model(x)
#                 loss = criterion(outputs, targets)

#                 loss.backward()
#                 optimizer.step()
#             print("Epoch end. Loss is:", loss.item())


#         # mlp_art_model = PyTorchClassifier(model=mlp_model, loss=criterion, optimizer=optimizer, input_shape=(24,), nb_classes=4)
#         return mlp_model

        
#     def bare_model(self):

#         model = MLPClassifierWrapper(solver='adam', hidden_layer_sizes=(32,16,8), random_state=1)

#         return model


# class NBModel(BaseModel):

#     model_name = 'Gaussian NB'

#     def train (self, dataset_train, SCALER):
#         print ('[INFO]: training Gaussian nb')
#         if SCALER:
#             model = make_pipeline(scaler, GaussianNB())
#         else:
#             model = make_pipeline(GaussianNB())
#         fit_params = {'gaussiannb__sample_weight': dataset_train.instance_weights}
#         model.fit(dataset_train.features, dataset_train.labels.ravel(), **fit_params)

#         return model

#     def bare_model(self):

#         model = GaussianNB()

#         return model


class TModel():

    def __init__(self, DATASET, model_type):
        self.DATASET = DATASET
        self.model_type = model_type
        print("This is how I see model type: ", self.model_type)

    def set_model(self, dataset, SCALER, ATTACK):
        if self.model_type == 'lr':
            model = LRModel() 
        elif self.model_type == 'rf':
            print("INSIDE RF")
            model = RFModel() 
        elif self.model_type == 'svm':
            model = SVMModel() 
        # elif self.model_type == 'nn':
        #     model = NNModel() 
        # elif self.model_type == 'nb':
        #     model = NBModel() 
        elif self.model_type == 'dt':
            model = DTModel() 
        elif self.model_type == 'dplr':
            model = DPLRModel()
        elif self.model_type == 'dprf':
            model = DPRF()
        elif self.model_type == 'mlp':
            model = MLP()
        trained_model = model.train(self.DATASET, dataset, SCALER, ATTACK)

        return trained_model


    def get_model(self):
        if self.model_type == 'lr':
            model = LRModel() 
        elif self.model_type == 'rf':
            model = RFModel() 
        elif self.model_type == 'svm':
            model = SVMModel() 
        # elif self.model_type == 'nn':
        #     model = NNModel() 
        # elif self.model_type == 'nb':
        #     model = NBModel() 
        elif self.model_type == 'dt':
            model = DTModel() 
        elif self.model_type == 'dplr':
            model = DPLRModel()
        elif self.model_type == 'dprf':
            model = DPRF()
        elif self.model_type == 'mlp':
            model = MLP()
        returned_model = model.bare_model()

        return returned_model


class MLPClassifierWrapper(MLPClassifier):

    def resample_with_replacement(self, X_train, y_train, sample_weight):

        # normalize sample_weights if not already
        #sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)
        sample_weight = sample_weight / np.sum(sample_weight.values,dtype=np.float64)
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()

        #X_train_resampled = np.zeros((len(X_train), len(X_train[0])), dtype=np.float32)
        X_train_resampled = np.zeros((X_train.shape), dtype=np.float32)
        #y_train_resampled = np.zeros((len(y_train)), dtype=np.int)
        y_train_resampled = np.zeros((y_train.shape), dtype=np.int)
        for i in range(X_train.shape[0]):
            # draw a number from 0 to len(X_train)-1
            draw = np.random.choice(np.arange(X_train.shape[0]), p=sample_weight)

            # place the X and y at the drawn number into the resampled X and y
            X_train_resampled[i] = X_train[draw]
            y_train_resampled[i] = y_train[draw]

        return X_train_resampled, y_train_resampled


    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            X, y = self.resample_with_replacement(X, y, sample_weight)

        return self._fit(X, y, incremental=(self.warm_start and
                                            hasattr(self, "classes_")))


