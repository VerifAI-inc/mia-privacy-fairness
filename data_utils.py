import numpy as np
import random
from imblearn.over_sampling import ADASYN

# Datasets
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset, LawSchoolGPADataset
from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21
from aif360.datasets import BinaryLabelDataset, StandardDataset

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
            import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
            
import pandas as pd

from sklearn.preprocessing import LabelEncoder

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
            
            df = pd.read_csv("./data/bank_reduced_dup.csv", index_col=0)

            # Create a BinaryLabelDataset using the binary labels ('y') and relevant attributes
            dataset = BinaryLabelDataset(
                favorable_label=1,  
                unfavorable_label=0,  
                df=df,
                label_names=['y'],  
                protected_attribute_names=['age']
            )
            # dataset = StandardDataset(
            #     df=df,
            #     label_name='y',  # Corrected: label_name should be a string
            #     favorable_classes=[1],  # Corrected: match the data type of the labels
            #     protected_attribute_names=['age']  # The protected attribute
            #     # privileged_classes=[lambda x: x >= 25]  # Assuming 'age' is numeric
            # )

            # dataset = BankDataset(
            #     protected_attribute_names=['age'],           # this dataset also contains protected
            #     privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
            # )
            
        elif self.DATASET == 'law_gender_aif':
            self.privileged_groups = [{'gender': 1}]
            self.unprivileged_groups = [{'gender': 0}]
            
            def custom_preprocessing(df):
                """Custom preprocessing to convert categorical features to numeric and create classification labels."""
                # Map gender: male = 1, female = 0
                df['gender'] = df['gender'].map({'male': 1, 'female': 0})
                
                # Map race: white = 1, black = 0
                df['race'] = df['race'].map({'white': 1, 'black': 0})
                
                # Create binary classification labels based on GPA
                gpa_threshold = 0.6  # You can adjust the threshold as needed
                df['gpa_class'] = (df['zfygpa'] >= gpa_threshold).astype(int)  # 1 for high GPA, 0 for low GPA
                
                df = df.drop(columns=['zfygpa'])
                
                return df

            # Load dataset and preprocess for classification
            reg_dataset = LawSchoolGPADataset(
                protected_attribute_names=['gender'],  # Use 'gender' as the protected attribute
                privileged_classes=[[1]],              # 'male' is the privileged class
                dep_var_name='gpa_class',              # Use 'gpa_class' as the new target variable for classification
                custom_preprocessing=custom_preprocessing  # Apply custom preprocessing
            )
            
            # Now, convert the dataset to a pandas DataFrame
            df, _ = reg_dataset.convert_to_dataframe()

            # Create a BinaryLabelDataset using the binary labels (gpa_class) and relevant attributes
            dataset = BinaryLabelDataset(
                favorable_label=1,  # 1 indicates "high GPA" (favorable outcome)
                unfavorable_label=0,  # 0 indicates "low GPA" (unfavorable outcome)
                df=df,
                label_names=['gpa_class'],  # The newly created binary label
                protected_attribute_names=['gender']  # The protected attribute (e.g., gender)
            )
            
        elif self.DATASET == 'law_race_aif':
            self.privileged_groups = [{'race': 1}]
            self.unprivileged_groups = [{'race': 0}]
            
            def custom_preprocessing(df):
                """Custom preprocessing to convert categorical features to numeric and create classification labels."""
                # Map gender: male = 1, female = 0
                df['gender'] = df['gender'].map({'male': 1, 'female': 0})
                
                # Map race: white = 1, black = 0
                df['race'] = df['race'].map({'white': 1, 'black': 0})
                
                # Create binary classification labels based on GPA
                gpa_threshold = 0.6  # You can adjust the threshold as needed
                df['gpa_class'] = (df['zfygpa'] >= gpa_threshold).astype(int)  # 1 for high GPA, 0 for low GPA
                
                df = df.drop(columns=['zfygpa'])
                
                return df

            # Load dataset and preprocess for classification
            reg_dataset = LawSchoolGPADataset(
                protected_attribute_names=['race'],  # Use 'race' as the protected attribute
                privileged_classes=[[1]],              # 'male' is the privileged class
                dep_var_name='gpa_class',              # Use 'gpa_class' as the new target variable for classification
                custom_preprocessing=custom_preprocessing  # Apply custom preprocessing
            )
            
            # Now, convert the dataset to a pandas DataFrame
            df, _ = reg_dataset.convert_to_dataframe()

            # Create a BinaryLabelDataset using the binary labels (gpa_class) and relevant attributes
            dataset = BinaryLabelDataset(
                favorable_label=1,  # 1 indicates "high GPA" (favorable outcome)
                unfavorable_label=0,  # 0 indicates "low GPA" (unfavorable outcome)
                df=df,
                label_names=['gpa_class'],  # The newly created binary label
                protected_attribute_names=['race']  # The protected attribute (e.g., gender)
            )
            
        elif self.DATASET == 'law_sex':
            self.privileged_groups = [{'gender': 1}]
            self.unprivileged_groups = [{'gender': 0}]
            
            df = pd.read_csv("./data/law_preprocessed.csv")

            # Create a BinaryLabelDataset using the binary labels (gpa_class) and relevant attributes
            dataset = BinaryLabelDataset(
                favorable_label=1,  # 1 indicates "high GPA" (favorable outcome)
                unfavorable_label=0,  # 0 indicates "low GPA" (unfavorable outcome)
                df=df,
                label_names=['pass_bar'],  # The newly created binary label
                protected_attribute_names=['gender']  # The protected attribute (e.g., gender)
            )
        
        elif self.DATASET == 'law_race':
            self.privileged_groups = [{'race': 1}]
            self.unprivileged_groups = [{'race': 0}]
            
            df = pd.read_csv("./data/law_preprocessed.csv")

            # Create a BinaryLabelDataset using the binary labels (gpa_class) and relevant attributes
            dataset = BinaryLabelDataset(
                favorable_label=1,  
                unfavorable_label=0, 
                df=df,
                label_names=['pass_bar'],  # The newly created binary label
                protected_attribute_names=['race']  # The protected attribute (e.g., black race)
            )
            
        elif self.DATASET == 'compas':
            self.privileged_groups = [{'race': 1}]
            self.unprivileged_groups = [{'race': 0}]
            dataset = load_preproc_data_compas()

        elif self.DATASET == 'german_age':
            self.privileged_groups = [{'age': 1}]
            self.unprivileged_groups = [{'age': 0}]
            
            def custom_preprocessing(df):
                credit_map = {1: 1.0, 2: 0.0}
                df['credit'] = df['credit'].replace(credit_map)
                
                print(df['credit'])
                
                return df
            
            dataset = GermanDataset(
                protected_attribute_names=['age'],           # this dataset also contains protected
                                                                # attribute for "sex" which we do not
                                                                # consider in this evaluation
                privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
                features_to_drop=['personal_status','sex'], # ignore sex-related attributes
                custom_preprocessing=custom_preprocessing
            )
        
        elif self.DATASET == 'german_foreign':
            self.privileged_groups = [{'foreign': 1}]
            self.unprivileged_groups = [{'foreign': 0}]
            
            
            def default_preprocessing(df):
                """Adds a derived sex attribute based on personal_status."""
                # TODO: ignores the value of privileged_classes for 'sex'
                #status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                #              'A92': 'female', 'A95': 'female'}
                #df['sex'] = df['personal_status'].replace(status_map)

                status_map = {'A201': 'Yes', 'A202': 'No'}
                df['foreign'] = df['foreign_worker'].replace(status_map)
                
                credit_map = {1: 1.0, 2: 0.0}
                df['credit'] = df['credit'].replace(credit_map)

                return df
            
            def custom_preprocessing(df):
                
                
                print(df['credit'])
                
                return df

            default_mappings = {
                'label_maps': [{1.0: 'Good Credit', 0.0: 'Bad Credit'}],
                'protected_attribute_maps': [{1.0: 'No', 0.0: 'Yes'}]
            }

            categorical_features=['status', 'credit_history', 'purpose',
                                    'savings', 'employment', 'other_debtors', 'property',
                                    'installment_plans', 'housing', 'skill_level', 'telephone']

            
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

    # Now apply ADASYN oversampling
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
    
    # print(f"Type of instance_weights: {type(f_dataset.instance_weights)}")
    # print(f"Shape of instance_weights: {getattr(f_dataset.instance_weights, 'shape', 'N/A')}")
    # print(f"Content of instance_weights: {f_dataset.instance_weights}")

    # print(f"Type of protected_attributes: {type(f_dataset.protected_attributes)}")
    # print(f"Shape of protected_attributes: {getattr(f_dataset.protected_attributes, 'shape', 'N/A')}")
    # print(f"Content of protected_attributes: {f_dataset.protected_attributes}")

    # Convert to lists if necessary
    instance_weights_list = f_dataset.instance_weights.flatten().tolist() if isinstance(f_dataset.instance_weights, np.ndarray) else f_dataset.instance_weights
    protected_attributes_list = f_dataset.protected_attributes.flatten().tolist() if isinstance(f_dataset.protected_attributes, np.ndarray) else f_dataset.protected_attributes

    # set weights and protected_attributes for the newly generated samples
    inc = X.shape[0]-f_dataset.features.shape[0]
    new_weights = [random.choice(instance_weights_list) for _ in range(inc)]
    new_attributes = [random.choice(protected_attributes_list) for _ in range(inc)]
    
    # new_attributes is 1D, reshape it to match the shape (n, 1)
    new_attributes = np.array(new_attributes).reshape(-1, 1)

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
    new_weights = [random.choice(instance_weights_list) for _ in range(inc)]
    new_attributes = [random.choice(protected_attributes_list) for _ in range(inc)]
    
    # new_attributes is 1D, reshape it to match the shape (n, 1)
    new_attributes = np.array(new_attributes).reshape(-1, 1)

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