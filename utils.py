# this is the python util file where I will create my different classes

# import all necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

# create a class for the training of the different models with the structure below:

# 1 Main class where the model data is loaded appropriately
    # 1a Method of the main class where the model is tokenized
#2 Sub-Class where we do the training of the unsupervised model
    # 2a Method where any ensemble methods, improvements are made to the model training
#3 Method where we predict the test data and measure accuracy
#4 Sub-class where we do a training of the supervised model
    # 4a Method where we improve training of the supervised model
# eventually, the supervised model should use the same method we defined for the prediction in order to test its accuracy

#this is the defined class for the models
class Articles():
    def __init__(self, data_train, data_test):
        # this class takes an input of the train data, and an input of the test data
        # both should be a dataframe
        self.data_train = data_train
        self.train_row_to_id = dict(enumerate(data_train['ArticleId']))
        self.data_test = data_test
        self.test_row_to_id = dict(enumerate(data_test['ArticleId']))
    
    def encode_labels(self, source: str = 'train'):
        """
        Factorize the data (train or test) and create consistent label-to-category mapping.
        """
        if source == 'train':
            data = self.data_train
            id_items, cat_items = pd.factorize(data['Category'])

            # Apply category IDs
            data['category_id'] = id_items
            self.factorized_train = data

            # Stable mapping using pandas Categorical
            # keep the exact order that factorize used so IDs stay consistent
            self.id_to_cat     = dict(enumerate(cat_items))               # 0→'business',1→'tech',…
            self.cat_to_id     = {v: k for k, v in self.id_to_cat.items()}

        elif source == 'test':
            data = self.data_test
            self.factorized_test = data

        else:
            raise ValueError("source must be either 'train' or 'test'")

    def vectorize_words(self, encoded_data):
        """
        this method will tokenize input dataframe with encoded labels to facilitate model training.
        It takes advantage of sklearn tokenizer. Note that this method expects the encoded_data to contain
        a column called Text with the articles of the data
        """
        if 'Text' in encoded_data.columns:
            tokenizer_model = TfidfVectorizer(sublinear_tf=True, max_features = 5000, max_df = 0.9, 
                                              min_df = 3, norm = 'l2', encoding ='latin-1', ngram_range=(1,2),
                                              stop_words = 'english')
            # get the fitted model
            # this is a matrix of size: # of artricles, # of words 
            self.tokenizer_model = tokenizer_model
            word_model = tokenizer_model.fit_transform(encoded_data.Text)
            self.word_tokens_sparse = word_model
            # return a dataframe with all of the information in one place
            summary_tokens = pd.DataFrame(self.word_tokens_sparse.toarray(), columns = self.tokenizer_model.get_feature_names_out())
            summary_tokens['category_id'] = self.factorized_train['category_id']
            summary_tokens['ArticleId'] = self.factorized_train['ArticleId']
            return summary_tokens

        else:
            raise ValueError("source contain column with Text title that contains text data")
    
    def NMF_execute(self, token_model_sparse, n_components=5, init='nndsvd', solver='cd',
                    random=42,alpha_W=0.0, alpha_H='same', l1_ratio=0.0, max_iter=200):
        """
        This method creates and fits NMF model given the sparse matrix self.word_model_sparse created
        with the vectorize_words() method. It returns the W and H components that the sparse matrix is
        decomposed into. The rest of the inputs are hyperparameters that can be modified with the creation
        of the sklearn model
        """
        summary_NNM = token_model_sparse.copy()

        #create the model and user hyperparameters provided in the method
        model_NMF = NMF(n_components=n_components, init=init, solver=solver, random_state=random,
                        alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio, max_iter=max_iter)
        self.model_NMF = model_NMF
        # fit the model with token data and get the W&H matrices
        W = model_NMF.fit_transform(summary_NNM)
        H = model_NMF.components_
        return W, H
    
    def label_permute_compare(self, ytdf, yp, n=5):
        """
    #     ytdf: labels dataframe object. These are the true labels
    #     yp: NMF label prediction output. a numpy array containing the index of
    #     the label with the highest score from the W matrix in the NMF_execute method
    #     Returns permuted label order and accuracy. 
    #     Example output: (3, 4, 1, 2, 0), 0.74 
    #     """
        label_permutation = itertools.permutations(range(n))
        unique_labels = ytdf['category_id'].unique()
        
        best_perm = None
        best_acc = 0
        for perm in label_permutation:
            # Create mapping from predicted cluster IDs to true category IDs
            key = dict(zip(range(n), perm))  # Map cluster 0->perm[0], cluster 1->perm[1], etc.
            
            # Map the predicted labels using this permutation
            yp_mapped = pd.Series(yp).map(key)  # Convert to Series for .map() method
            
            accuracy = accuracy_score(ytdf['category_id'], yp_mapped)
            if accuracy > best_acc:
                best_acc = accuracy
                best_perm = perm
        return best_perm, best_acc

    def confusion_matrix_mine(self, ytdf,yp, labelorder):
        """
        ytdf: labels dataframe object. These are the true labels
        yp: NMF label prediction output. a numpy array containing the index of
        the label with the highest score from the W matrix in the NMF_execute method
        labelorder is the best_perm parameter returned from the label_permute_compare method
        """
        cat_types = self.id_to_cat.values()
        # Create a mapping from category IDs to the order in which they appear in labelorder
        # This ensures that the confusion matrix rows/columns are in the same order as the original categories
        # Invert labelorder so each true category ID points to its cluster index
        key = {cat_id: idx for idx, cat_id in enumerate(labelorder)}
        ytrue_mapped = ytdf['category_id'].map(key)
        cm = confusion_matrix(ytrue_mapped,yp)
        # print(cm)
        return cm

# class for the development of the supervised model
class SupervisedArticles(Articles):
    def __init__(self, data_train, data_test):
        # this class inherits from Articles and is used for supervised model training
        # it takes an input of the train data, and an input of the test data
        # both should be a dataframe
        # call the parent class constructor
        # and encodes the labels for both train and test data
        # then vectorizes the words in the train data
        # and stores the sparse matrix in self.train_tokens
        # and the factorized train data in self.factorized_train
        # and the tokenizer model in self.tokenizer_model
        # and the id_to_cat mapping in self.id_to_cat
        # and the cat_to_id mapping in self.cat_to_id
        super().__init__(data_train, data_test)
        self.encode_labels(source='train')
        self.encode_labels(source='test')
        self.train_tokens = self.vectorize_words(self.factorized_train)
    
    def split_train_validation(self, validation_size=0.2):
        """
        This method splits the train data into train and validation sets.
        It returns the train and validation dataframes.
        """
        # Shuffle the train data
        shuffled_data = self.factorized_train.sample(frac=1, random_state=42).reset_index(drop=True)
        # Calculate the split index
        split_index = int(len(shuffled_data) * (1 - validation_size))
        # Split the data
        train_data = shuffled_data[:split_index]
        validation_data = shuffled_data[split_index:]
        return train_data, validation_data

    def train_random_forest (self, train_data_vectorized, validation_data_vectorized, 
                             train_data_y, validation_data_y):
        """
        This method trains a random forest model on the train data and selects the best hyperparameters
        using the validation data. It returns the trained model and the best hyperparameters.
        """
        # this method trains a random forest model on the train data
        # it also selects the best hyperparameters for the model using the validation data
        # train the random forest model with the train data
        clf = RandomForestClassifier(max_depth=2, random_state=42)
        clf.fit(train_data_vectorized, train_data_y)
        self.trained_random_forest = clf

        # predict the accuracy with the trained data
        trained_predictions = clf.predict(train_data_vectorized)
        acc_trained = accuracy_score(train_data_y, trained_predictions)
        
        # do hypperparameter tuning with the validation data
        # create a grid search for the hyperparameters
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_features': ['auto', 'sqrt', 'log2'], 
            'bootstrap': [True, False],
            'ccp_alpha': [0.0, 0.01, 0.1, 0.5, 1.0]
            }
        # create tge girs search object
        clf_validation = RandomForestClassifier(random_state=42)
        # do the grid serach action
        grid_search = GridSearchCV(clf_validation, param_grid, cv=3, n_jobs=-1, verbose=2,scoring='accuracy')
        # fit the grid search with the validation data
        grid_search.fit(validation_data_vectorized, validation_data_y)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        # predict the accuracy with the validation data
        validation_predictions = best_model.predict(validation_data_vectorized)
        acc_validation = accuracy_score(validation_data_y, validation_predictions)
        self.best_model_random_forest = best_model
        self.best_parameters_random_forest = best_params
        return acc_trained, acc_validation

    def train_linear_SVM(self, train_data_vectorized, validation_data_vectorized, 
                         train_data_y, validation_data_y):
        """this method uses the vectorized and factorized train data to train a linear SVM model.
        As an input it takes the train data and the validation data vectorized that the split_train_validation
        method returns. It uses the train data to train the model and the validation data to select the best hyperparameters.
        It returns the trained model and the best hyperparameters.
        """
        # create initial plain nodel and use standard scaler to scale the data appropriately
        # since linear SVM is sensitive to the scale of the data
        clf = Pipeline(steps=[('standardscaler', StandardScaler(with_mean=False)),
                ('linearsvc', LinearSVC(random_state=0, tol=1e-05))])
        
        # fit the model with the train data
        clf.fit(train_data_vectorized, train_data_y)
        self.trained_linear_SVM = clf
        # predict the accuracy with the trained data
        trained_predictions = clf.predict(train_data_vectorized)
        acc_trained = accuracy_score(train_data_y, trained_predictions)
        
        # evaluate the model with the validation data
        val_predictions = clf.predict(validation_data_vectorized)
        acc_val = accuracy_score(validation_data_y, val_predictions)
        
        # do hyperparameter tuning with the validation data
        # create a grid search for the hyperparameters
        param_grid = {
            'linearsvc__C': [0.1, 1, 10, 100],
            'linearsvc__max_iter': [1000, 2000, 3000],
            'linearsvc__loss': ['hinge', 'squared_hinge'],
            'linearsvc__dual': [True, False]
        }
        # create the grid search object
        grid_search = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
        # fit the grid search with the validation data
        grid_search.fit(train_data_vectorized, train_data_y)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        # use the best model to predict the accuracy with the validation data
        validation_predictions_best_model = best_model.predict(validation_data_vectorized)
        acc_val_best_model = accuracy_score(validation_data_y, validation_predictions_best_model)

        return acc_trained, acc_val, acc_val_best_model, best_params, best_model

