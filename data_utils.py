import numpy as np
import random
from imblearn.over_sampling import ADASYN

# Datasets
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset
from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21
from aif360.datasets import BinaryLabelDataset


from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
            import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas

class DatasetBuilder:

    def __init__(self, DATASET):
        self.DATASET = DATASET
        self.privileged_groups =[]
        self.unprivileged_groups=[]

    def print_dataset_name(self):
        print(self.DATASET)

    # load data 'adult', 'grade', 'bank', 'german', 'compas', or 'meps'
    def load_data(self):
        if self.DATASET == 'adult':
            protected_attribute_used = 1

            if protected_attribute_used == 1:
                self.privileged_groups = [{'sex': 1}]
                self.unprivileged_groups = [{'sex': 0}]
                dataset = load_preproc_data_adult(['sex'])
            else:
                self.privileged_groups = [{'race': 1}]
                self.unprivileged_groups = [{'race': 0}]
                dataset = load_preproc_data_adult(['race'])

        elif self.DATASET == 'bank':
            self.privileged_groups = [{'age': 1}]
            self.unprivileged_groups = [{'age': 0}]
            dataset = BankDataset(
                protected_attribute_names=['age'],           # this dataset also contains protected
                privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
            )

        elif self.DATASET == 'compas':
            self.privileged_groups = [{'race': 1}]
            self.unprivileged_groups = [{'race': 0}]
            dataset = load_preproc_data_compas()

        elif self.DATASET == 'german':
            # 1:age ,2: foreign
            protected_attribute_used = 1
            if protected_attribute_used == 1:
                self.privileged_groups = [{'age': 1}]
                self.unprivileged_groups = [{'age': 0}]
                dataset = GermanDataset(
                    protected_attribute_names=['age'],           # this dataset also contains protected
                                                                 # attribute for "sex" which we do not
                                                                 # consider in this evaluation
                    privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
                    features_to_drop=['personal_status','sex'], # ignore sex-related attributes
                )
            else:
                self.privileged_groups = [{'foreign': 1}]
                self.unprivileged_groups = [{'foreign': 0}]

                default_mappings = {
                    'label_maps': [{1.0: 'Good Credit', 2.0: 'Bad Credit'}],
                    'protected_attribute_maps': [{1.0: 'No', 0.0: 'Yes'}]
                }

                categorical_features=['status', 'credit_history', 'purpose',
                                     'savings', 'employment', 'other_debtors', 'property',
                                     'installment_plans', 'housing', 'skill_level', 'telephone']

                def default_preprocessing(df):
                    """Adds a derived sex attribute based on personal_status."""
                    # TODO: ignores the value of privileged_classes for 'sex'
                    #status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                    #              'A92': 'female', 'A95': 'female'}
                    #df['sex'] = df['personal_status'].replace(status_map)

                    status_map = {'A201': 'Yes', 'A202': 'No'}
                    df['foreign'] = df['foreign_worker'].replace(status_map)

                    return df

                dataset = GermanDataset(
                    protected_attribute_names=['foreign'],       # this dataset also contains protected
                                                                 # attribute for "sex" which we do not
                                                                 # consider in this evaluation
                    privileged_classes=[['No']],                 # none foreign is considered privileged
                    features_to_drop=['personal_status', 'foreign_worker'], # ignore sex-related attributes
                    categorical_features=categorical_features,
                    custom_preprocessing=default_preprocessing,
                    metadata=default_mappings
                )

        elif self.DATASET == 'grade':
            #load dataset and print shape
            dataset_loc = "./student/student-por.csv"
            df = pd.read_csv(dataset_loc, sep=";")
            print('Dataset consists of {} Observations and {} Variables'.format(df.shape[0],df.shape[1]))
            df.drop(['G1', 'G2'], inplace=True, axis=1)
            features = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                   'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
                   'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                   'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
                   'Walc', 'health']
            labels = ['absences', 'G3']
            df['sex'] = df['sex'].map( {'F':0, 'M':1})
            df['Pstatus'] = df['Pstatus'].map( {'A':0, 'T':1})
            df['age'].values[df['age'] < 18] = 0
            df['age'].values[df['age'] >= 18] = 1
            df['G3'].values[df['G3'] < 12] = 0
            df['G3'].values[df['G3'] >= 12] = 1
            df['G3'].unique()
            df['absences'].values[df['absences'] <= 4] = 1
            df['absences'].values[df['absences'] > 4] = 0
            df['absences'].unique()

            numvar = [key for key in dict(df.dtypes)
                               if dict(df.dtypes)[key]
                                   in ['float64','float32','int32','int64']] # Numeric Variable

            catvar = [key for key in dict(df.dtypes)
                         if dict(df.dtypes)[key] in ['object'] ] # Categorical Varible
            for cat in catvar:
                df[cat] = LabelEncoder().fit_transform(df[cat])

            proclsvars = ['sex', 'Pstatus', 'age']
            depenvars = ['G3', 'absences']

            proclsvar = 'sex'
            depenvar = 'G3'

            self.privileged_groups = [{proclsvar: 1}]
            self.unprivileged_groups = [{proclsvar: 0}]
            favorable_label = 0
            unfavorable_label = 1
            dataset = BinaryLabelDataset(favorable_label=favorable_label,
                                unfavorable_label=unfavorable_label,
                                df=df,
                                label_names=[depenvar],
                                protected_attribute_names=[proclsvar],
                                unprivileged_protected_attributes=self.unprivileged_groups)
        else:
            if self.DATASET == 'meps19':
                dataset = MEPSDataset19()
            elif self.DATASET == 'meps20':
                dataset = MEPSDataset20()
            else:
                dataset = MEPSDataset21()

            sens_ind = 0
            sens_attr = dataset.protected_attribute_names[sens_ind]
            self.unprivileged_groups = [{sens_attr: v} for v in
                                        dataset.unprivileged_protected_attributes[sens_ind]]
            self.privileged_groups = [{sens_attr: v} for v in
                                      dataset.privileged_protected_attributes[sens_ind]]


        return dataset


#balance dataset by synthetically generate instances
# 1. First, inflate the uf_label group for oversampling purpose
# 2. Next, generate "n_extra" samples with "f_label"
# 3. Return expanded dataset as a whole and the extra set separately

def balance(dataset, n_extra, inflate_rate, f_label, uf_label):

    # make a duplicate copy of the input data
    dataset_transf_train = dataset.copy(deepcopy=True)

    # subsets with favorable labels and unfavorable labels
    f_dataset = dataset.subset(np.where(dataset.labels==f_label)[0].tolist())
    uf_dataset = dataset.subset(np.where(dataset.labels==uf_label)[0].tolist())

    # expand the group with uf_label for oversampling purpose
    inflated_uf_features = np.repeat(uf_dataset.features, inflate_rate, axis=0)
    sample_features = np.concatenate((f_dataset.features, inflated_uf_features))
    inflated_uf_labels = np.repeat(uf_dataset.labels, inflate_rate, axis=0)
    sample_labels = np.concatenate((f_dataset.labels, inflated_uf_labels))

    # oversampling favorable samples
    # X: inflated dataset with synthetic samples of f_label attached to the end 
    oversample = ADASYN(sampling_strategy='minority')
    X, y = oversample.fit_resample(sample_features, sample_labels)
    y = y.reshape(-1,1)

    # take samples from dataset with only favorable labels
    X = X[np.where(y==f_label)[0].tolist()]  # data with f_label + new samples
    y = y[y==f_label]

    selected = int(f_dataset.features.shape[0]+n_extra)

    X = X[:selected, :]
    y = y[:selected]
    y = y.reshape(-1,1)

    # set weights and protected_attributes for the newly generated samples
    inc = X.shape[0]-f_dataset.features.shape[0]
    new_weights = [random.choice(f_dataset.instance_weights) for _ in range(inc)]
    new_attributes = [random.choice(f_dataset.protected_attributes) for _ in range(inc)]

    # compose transformed dataset
    dataset_transf_train.features = np.concatenate((uf_dataset.features, X))
    dataset_transf_train.labels = np.concatenate((uf_dataset.labels, y))
    dataset_transf_train.instance_weights = np.concatenate((uf_dataset.instance_weights, f_dataset.instance_weights, new_weights))
    dataset_transf_train.protected_attributes = np.concatenate((uf_dataset.protected_attributes, f_dataset.protected_attributes, new_attributes))

    # make a duplicate copy of the input data
    dataset_extra_train = dataset.copy()

    X_ex = X[-int(n_extra):]
    y_ex = y[-int(n_extra):]
    y_ex = y_ex.reshape(-1,1)

    # set weights and protected_attributes for the newly generated samples
    inc = int(n_extra)
    new_weights = [random.choice(f_dataset.instance_weights) for _ in range(inc)]
    new_attributes = [random.choice(f_dataset.protected_attributes) for _ in range(inc)]

    # compose extra dataset
    dataset_extra_train.features = X_ex
    dataset_extra_train.labels = y_ex
    dataset_extra_train.instance_weights = new_weights
    dataset_extra_train.protected_attributes = new_attributes

    # verifying
    #print(dataset_transf_train.features.shape)
    #print(dataset_transf_train.labels.shape)
    #print(dataset_transf_train.instance_weights.shape)
    #print(dataset_transf_train.protected_attributes.shape)

    # return favor and unfavored oversampling results
    return dataset_transf_train, dataset_extra_train


