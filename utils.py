# this is the python util file where I will create my different classes

# import all necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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
        self.data_test = data_test
    
    def encode_labels(self, source: str = 'train'):
        """
        Factorize the data that you are going to be working with to ease with the model building
        This method takes the created data frame for the train or test sets and it factorizes it
        """
        # Figure out whether to do the factorization with train or with test
        if source ==  'train':
            data = self.data_train
        elif source == 'test':
            data  =self.data_test
        else:
            raise ValueError("source must be either train or test")
        # factorize the categories in each data frame
        id_items, cat_items= data.Category.factorize()
        data['category_id'] = id_items
        
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
        if 'Text' in df_train.columns:
            tokenizer_model = TfidfVectorizer(sublinear_tf=True, min_df = 3, norm = 'l2', encoding ='latin-1', ngram_range=(1,2),
                                              stop_words = 'english')
            # get the fitted model
            word_model = tokenizer_model.fit_transform(encoded_data.Text).toarray()
            self.word_tokens = word_model
        else:
            raise ValueError("source contain column with Text title that contains text data")