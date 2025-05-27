# this is the python util file where I will create my different classes

# import all necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools

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
        Factorize the data that you are going to be working with to ease with the model building
        This method takes the created data frame for the train or test sets and it factorizes it
        """
        # Figure out whether to do the factorization with train or with test
        if source ==  'train':
            data = self.data_train
            # factorize the categories in each data frame
            id_items, cat_items= data.Category.factorize()
            data['category_id'] = id_items
        elif source == 'test':
            data  =self.data_test
        else:
            raise ValueError("source must be either train or test")
        if source == 'train':
            self.factorized_train = data
            #create a dictionary with the factorization
            #filter the ids and categories to not have repetition
            #only create the data map with the train dataset
            track_item = set()
            id_list = [id for id in id_items if id not in track_item and not track_item.add(id)]
            track_cat = set()
            cat_list = [cat for cat in cat_items if cat not in track_cat and not track_cat.add(cat)]
            self.category_to_id = dict(zip(id_list, track_cat))
        else:
            # source == 'test'
            self.factorized_test = data
    def vectorize_words(self, encoded_data):
        """
        this method will tokenize input dataframe with encoded labels to facilitate model training.
        It takes advantage of sklearn tokenizer. Note that this method expects the encoded_data to contain
        a column called Text with the articles of the data
        """
        if 'Text' in encoded_data.columns:
            tokenizer_model = TfidfVectorizer(sublinear_tf=True, min_df = 3, norm = 'l2', encoding ='latin-1', ngram_range=(1,2),
                                              stop_words = 'english')
            # get the fitted model
            # this is a matrix of size: # of artricles, # of words 
            word_model = tokenizer_model.fit_transform(encoded_data.Text).toarray()
            self.word_tokens = word_model
            # get a version of the model that is a sparse matrix for further transformation
            self.word_model_sparse = tokenizer_model.fit_transform(encoded_data.Text)
            self.tokenizer_model = tokenizer_model
            # return a dataframe with all of the information in one place
            summary_tokens = pd.DataFrame(self.word_tokens, columns = self.tokenizer_model.get_feature_names_out())
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
        # fit the model with token data and get the W&H matrices
        W = model_NMF.fit_transform(summary_NNM)
        H = model_NMF.components_
        self.model_NMF = model_NMF
        return W, H
    
    def label_permute_compare(self, ytdf,yp,n=5):
        """
        ytdf: labels dataframe object. These are the true labels
        yp: NMF label prediction output. a numpy array containing the index of
        the label with the highest score from the W matrix in the NMF_execute method
        Returns permuted label order and accuracy. 
        Example output: (3, 4, 1, 2, 0), 0.74 
        """
        label_permutation = itertools.permutations(range(n))
        unique_labels = ytdf['category_id'].unique()
        #now associate a key of label for each permutation
        best_perm = None
        best_acc = 0
        for perm in label_permutation:
            key = dict(zip(unique_labels, perm))
            #map the key to the ytru data
            ytrue_mapped = ytdf['category_id'].map(key)
            accuracy = accuracy_score(ytrue_mapped, yp)
            if accuracy>best_acc:
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
        cat_types = ytdf['category_id'].unique()
        key = dict(zip(cat_types, labelorder))
        ytrue_mapped = ytdf['category_id'].map(key)
        cm = confusion_matrix(ytrue_mapped,yp)
        # print(cm)
        return cm

