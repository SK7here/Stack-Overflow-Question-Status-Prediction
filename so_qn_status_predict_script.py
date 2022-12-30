# Importing Packages
import numpy as np  
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import circlify
from wordcloud import WordCloud, STOPWORDS

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GRU, Dropout, Bidirectional, SpatialDropout1D

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# To allow assigning values to dataframe slices
pd.options.mode.chained_assignment = None

# To allow displaying all the columns of the dataframe
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



# Importing dataset
raw_data=pd.read_csv("train-sample.csv")

# Description on the dataset
print('Information about dataset\n')
raw_data.info()

# Checking null values in the dataset
print("\n\n")
print('Null values in the dataset\n')
print(raw_data.isna().sum())



'''
All the Tags that are present for a given post will be combined into one single Tag

Since an NLP model is being built for the prediction,
    -> 'Tags', 'title' and 'body' columns are considered as features
    -> 'OpenStatus' columns is the target column
'''
# Filtering the required columns
filtered_data = raw_data[['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5', 'Title', 'BodyMarkdown', 'OpenStatus']]

# Replacing null Tags with empty strings
filtered_data['Tag1'] = filtered_data['Tag1'].fillna('')
filtered_data['Tag2'] = filtered_data['Tag2'].fillna('')
filtered_data['Tag3'] = filtered_data['Tag3'].fillna('')
filtered_data['Tag4'] = filtered_data['Tag4'].fillna('')
filtered_data['Tag5'] = filtered_data['Tag5'].fillna('')

# Concatenating all the Tag columns into one single column
filtered_data['Tags'] = filtered_data[['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5']].agg(' '.join, axis=1)

# Dropping all the individial tag columns
filtered_data.drop(['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5'], axis=1, inplace=True)
print("\n\n")
print("After merging all the Tags of the post\n")
print(filtered_data.head(10))

# Label encoding target variable
print("\n\n")
print("Possible statuses of the StackOverflow post\n")
print(filtered_data.OpenStatus.unique())

labelencoder = LabelEncoder()
filtered_data['OpenStatus'] = labelencoder.fit_transform(filtered_data['OpenStatus'])
print("\nLabel encoder mappings")
labelencoder_mapping = {l: i for i, l in enumerate(labelencoder.classes_)}
print(labelencoder_mapping)

print("\n\n")
print("After performing label encoding\n")
print(filtered_data.head(10))

# Combining the columns 'Title', 'BodyMarkdown', 'Tags' into one column
filtered_data['Post'] = filtered_data[['Title', 'BodyMarkdown', 'Tags']].agg(' '.join, axis=1)
# Dropping the individual columns that were merged
filtered_data.drop(['Title', 'BodyMarkdown', 'Tags'], axis=1, inplace=True)
dataset = filtered_data[['Post', 'OpenStatus']]

print("\n\n")
print("Final dataset\n")
print(dataset.head(10))

####  dataset = dataset.sample(frac = 0.005)
print(dataset.shape)



# Visualization
stop_words = set(stopwords.words("english"))

#create function to get a DataFrame
def generate_word_count_df(text):
    # Create list of unique words
    words_list = text.split(' ')
    unique_words_list = list(set(words_list))
    
    #remove stop words, single letter strings and numbers
    filtered_set_words = [i for i in unique_words_list if ((i not in stop_words) and (i.isalnum()) and (len(i) > 1))]
    
    #count occurence of each word in the text
    words_count_list = [words_list.count(i) for i in filtered_set_words]
    
    #create DataFrame of word count
    wrd_cnt_df = pd.DataFrame(zip(filtered_set_words, words_count_list), columns=['word','count'])
    wrd_cnt_df.sort_values('count', ascending=False, inplace=True)
    wrd_cnt_df.reset_index(drop=True, inplace=True)
    return wrd_cnt_df

#create function to get a color dictionary
def generate_color_dictionary(palette, max_val, start):
    palette_obj = list(sns.color_palette(palette=palette, n_colors=max_val).as_hex())
    color_dictionary = dict(enumerate(palette_obj, start=start))
    return color_dictionary

word_dictionary = {}
word_cnt_dictionary = {}

word_dictionary['not a real question'] = ' '.join([text for text in dataset['Post'][dataset['OpenStatus'] == labelencoder_mapping['not a real question']]])
word_dictionary['not constructive'] = ' '.join([text for text in dataset['Post'][dataset['OpenStatus'] == labelencoder_mapping['not constructive']]])
word_dictionary['off topic'] = ' '.join([text for text in dataset['Post'][dataset['OpenStatus'] == labelencoder_mapping['off topic']]])
word_dictionary['open'] = ' '.join([text for text in dataset['Post'][dataset['OpenStatus'] == labelencoder_mapping['open']]])
word_dictionary['too localized'] = ' '.join([text for text in dataset['Post'][dataset['OpenStatus'] == labelencoder_mapping['too localized']]])


word_cnt_dictionary['not a real question'] = generate_word_count_df(word_dictionary['not a real question'])
word_cnt_dictionary['not constructive'] = generate_word_count_df(word_dictionary['not constructive'])
word_cnt_dictionary['off topic'] = generate_word_count_df(word_dictionary['off topic'])
word_cnt_dictionary['open'] = generate_word_count_df(word_dictionary['open'])
word_cnt_dictionary['too localized'] = generate_word_count_df(word_dictionary['too localized'])


# Word cloud of each class
for key, value in word_dictionary.items():
    wordcloud = WordCloud(width= 1000, height = 600, max_words=100,
                          random_state=1, background_color='gray', colormap='viridis_r',
                          collocations=False, stopwords = STOPWORDS).generate(value)
    plt.figure(figsize=(10,6))
    plt.imshow(wordcloud)
    plt.title(key, fontsize=15, loc='center')
    plt.axis("off")
    plt.show()

# Bar Chart Grid - top 20 words in each class
for key, value in word_cnt_dictionary.items():##    
    # Indices for slicing top 20 words into 4 sections for visulization purpose
    index_list = [[i[0],i[-1]+1] for i in np.array_split(range(20), 2)]

    max_count = value['count'].max()
    color_dict = generate_color_dictionary('viridis', max_count, 1)

    fig, axs = plt.subplots(1, 2, figsize=(10,6), facecolor='white', squeeze=False)
    for col, idx in zip(range(0,2), index_list):
        df = value[idx[0]:idx[-1]]
        label = [w + ': ' + str(n) for w,n in zip(df['word'],df['count'])]
        color_l = [color_dict.get(i) for i in df['count']]
        x = list(df['count'])
        y = list(range(0,10))
        
        sns.barplot(x = x, y = y, data=df, alpha=0.9, orient = 'h',
                    ax = axs[0][col], palette = color_l)
        step = max_count//3
        axs[0][col].set_xticks(list(range(0,max_count+1,step)))
        axs[0][col].set_yticklabels(label, fontsize=12)
        axs[0][col].spines['bottom'].set_color('white')
        axs[0][col].spines['right'].set_color('white')
        axs[0][col].spines['top'].set_color('white')
        axs[0][col].spines['left'].set_color('white')
                
    plt.tight_layout(pad=3.0)
    plt.title(key, fontsize=15)
    plt.show()

# Tree Map - top 50 words in each class
for key, value in word_cnt_dictionary.items():
    fig = px.treemap(value[0:50], path=[px.Constant(key), 'word'],
                     values='count',
                     color='count',
                     color_continuous_scale='viridis',
                     color_continuous_midpoint=np.average(value['count'])
                    )
    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    fig.show()

# Circle Packing - top 25 words in each class
for key, value in word_cnt_dictionary.items():
    # compute circle positions:
    circles = circlify.circlify(value['count'][0:25].tolist(), 
                                show_enclosure=False, 
                                target_enclosure=circlify.Circle(x=0, y=0)
                               )
    max_count = value['count'][0:25].max()
    color_dictionary = generate_color_dictionary('RdYlBu_r',max_count ,1)

    fig, ax = plt.subplots(figsize=(10,6), facecolor='white')
    ax.axis('off')
    lim = max(max(abs(circle.x)+circle.r, abs(circle.y)+circle.r,) for circle in circles)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    # list of labels
    labels = list(value['word'][0:25])
    counts = list(value['count'][0:25])
    labels.reverse()
    counts.reverse()

    # print circles
    for circle, label, count in zip(circles, labels, counts):
        x, y, r = circle
        ax.add_patch(plt.Circle((x, y), r, alpha=0.9, color = color_dictionary.get(count)))
        plt.annotate(label +'\n'+ str(count), (x,y), size=12, va='center', ha='center')
    plt.xticks([])
    plt.yticks([])
    plt.show()



# Train-Validation split
X_train, X_val, y_train, y_val = train_test_split(dataset['Post'], dataset['OpenStatus'], test_size=0.2)

# Changing each of the output label as a list
y_train_encoded = to_categorical(y_train)
y_val_encoded = to_categorical(y_val)

classes_count = y_train_encoded.shape[1]

# Number of words in the longest post
row_max_length = max([len(x.split()) for x in dataset['Post'].values])

# Tokenizer for the training dataset
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset['Post'].values)

# Number of unique words in training dataset
vocabulary_size = len(tokenizer.word_index) + 1

# convert words of train, validation and test data into integers
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_val_tokens = tokenizer.texts_to_sequences(X_val)

# Padding to make sure that all sequences are of same length
X_train_pad = pad_sequences(X_train_tokens, maxlen=row_max_length, padding='post')
X_val_pad = pad_sequences(X_val_tokens, maxlen=row_max_length, padding='post')

# Building the Deep Learning Model - Bidirectional Gated Recurrent Unit
'''
Bidirectional Gated recurrent models:
-> process sequneces with two GRUs
-> One of the GRUs takes input in forward direction (past to future) while the eother in backward (future to past)
-> have just two gates 'input' and 'forget'
-> Better than unidirectional LSTM - future information makes it easy to predict the word along with the past information
-> Does not need memory units; Hence easy to train
'''
EMBEDDING_DIM = 256

model = Sequential()
model.add(Embedding(vocabulary_size, EMBEDDING_DIM, input_length=row_max_length))
# In early layers there would be correlation among adjacent frames in the feature maps.
# Hence, to make sure that feature maps are independent, SpatialDropout1D is used that removes 1D featuremaps instead of individual elements
model.add(SpatialDropout1D(0.2))
# Bidirectional GRU layer
model.add(Bidirectional(GRU(128)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
# Softmax activation is used, as this a multiclass classification problem
model.add(Dense(classes_count, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(X_train_pad, y_train_encoded, epochs=20, validation_data=(X_val_pad, y_val_encoded), batch_size=128, callbacks=[callback], verbose=1)



# predict test data and generate performance metrics
y_pred_ = model.predict(X_val_pad)
y_pred = [np.argmax(x) for x in y_sub_hat_]

accuracy_metric = accuracy_score(y_val, y_pred)
# weighted average to tackle class imbalance
precision_metric = precision_score(y_val, y_pred, average='weighted')
recall_metric = recall_score(y_val, y_pred, average='weighted')
f1_score_metric = f1_score(y_val, y_pred, average='weighted')

print("\n\n")
print("Performance metrics for the model is: ")
print("\nAccuracy: {}" .format(accuracy_metric))
print("\nPrecision: {}" .format(precision_metric))
print("\nRecall: {}".format(recall_metric))
print("\nF1-Score: {}".format(f1_score_metric))     
