
# Update the VM with the latest and greatest
!pip install --upgrade pip
!pip install langid
!pip install pandas
!pip install nltk

# all the imports
from __future__ import absolute_import, division, print_function, unicode_literals

import langid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import re
import string

# Need to download stop words corpus
nltk.download('stopwords')
# Need to download punkt
nltk.download('punkt')

# Read the files into Pandas dataframes for processing
# data1 and data2 will contain 'pkgname' and 'Descriptions'
data1 = pd.read_csv('ContextualDataDescriptionWhatsNew-part1.csv')
data2 = pd.read_csv('ContextualDataDescriptionWhatsNew-part2.csv')
# data will contain the column 'ID', which is the same as 'pkgname' in data1,
# and other columns called 'Genr', 'Developer',
# plus 4 different permission scores.
# The column called 'permission_2' has the score for our study
data = pd.read_csv('apks_with_scores.csv')

# Temporary dataframes to hold only the columns we will use later on
df1=data1[['pkgname','Description']]
df2=data2[['pkgname','Description']]
df3=data[['ID','permission_2']]

# Perfoming the Merge and getting the class label
# ---

frames=[df1,df2]
df_cat=pd.concat(frames)

# Remove duplicates
df_cat_final=df_cat.drop_duplicates()

# Then we perform a merge using the 'pkgname' and 'ID' columns
df_cat_merge=pd.merge(df_cat_final,df3,left_on=['pkgname'], right_on=['ID'])

# Remove any rows containing NAs, to avoid processing problems
df_cat_merge = df_cat_merge.dropna()

# Save file for later processing
# First, make sure that all the 'Description' fields contain a string
df_cat_merge['Description'] = df_cat_merge['Description'].map(lambda x: str(x))
df_cat_merge.to_csv('CategoryMergeRaw.txt', sep='\t', index=False)

# We can drop the column 'ID', which was used for the match
# and should be a duplicate of 'pkgname'
df_cat_merge = df_cat_merge.drop(columns=['ID'])

# Dataframe update

# A new column called 'desclang' will contain a 2 letter ISO 639-1 code
# For our tests we are only interested in English, represented by 'en'
# We use langid.py to classify the apps according to the description language
##
# NOTE
##
# At this point we ran into problems with language classification due to app
# descriptions that contained multiple languages.
# We found that truncating the number of words provided to the language
# classifier provided much more accurate classification
trunc_words = 350

df_cat_merge = df_cat_merge.assign(
    desclang=df_cat_merge['Description'].apply(
        lambda row: langid.classify(' '.join(row.split()[:trunc_words]))[0]
        )
    )

# Once again, remove any NAs
df_cat_merge = df_cat_merge.dropna()

# Save file for later processing
df_cat_merge.to_csv('CategoryMergeLanguage2.txt', sep='\t', index=False)

# remove non-English apps
df_cat_merge = df_cat_merge.where(df_cat_merge['desclang'] == 'en')

# Save file for later processing
df_cat_merge.to_csv('CategoryMergeEnglish2.txt', sep='\t', index=False)

# Remove special characters
# NOTE: This will need to be tweaked for any languages other than English
nospace = lambda x: re.sub(
    r'[^a-zA-Z ]', # Regular expression removing non-alpha characters
    '',
    re.sub(
        r'[^a-zA-Z\` ]',  # Replace non-alpha characters for spaces
        ' ',
        re.sub(
            # Remove tabs, new line, carriage returns, form feeds,
            # and UNICODE characters
            r'(\\t|\\n|\\r|\\f|\\u[0-9a-fA-F]{4})',
            '',
            str(x)
            )
        )
    )

# Remove stop words
sw = set(nltk.corpus.stopwords.words('english'))
nostops = lambda x: ' '.join(
    str(word).lower() for word in nltk.tokenize.word_tokenize(x) if word not in sw
    )

# Putting it all together and removing extra spaces
cleanup = lambda x: re.sub(r'\s+', ' ', nostops(nospace(x)))

# All descriptions
# They should be, but let's make sure all 'Description' fields are strings
df_cat_merge['Description'] = df_cat_merge['Description'].map(lambda x: str(x))

# Save pre-processed data for later use
df_cat_merge.to_csv('CategoryMergePreProcessed2.txt', sep='\t', index=False)

# Prepare the Snowball Stemmer
stemmer = nltk.stem.SnowballStemmer('english')

df_cat_merge['Description'] = df_cat_merge['Description'].map(
    lambda x: ' '.join(stemmer.stem(word) for word in x.split())
    )

# Save pre-processed data
df_cat_merge.to_csv('CategoryMergePreProcessedStemming2.txt', sep='\t', index=False)

# Take Genre data from original file above
context_data = data[['ID','Genr']]
# Make sure all Genr data is a string
context_data['Genr'] = context_data['Genr'].map(lambda x: str(x))
# Remove blanks
context_data = context_data.dropna()

# Remove ampersands from Genres
context_data['Genr'] = context_data['Genr'].map(lambda x: re.sub(r'&', '', x))
# Use stemmer on the Genre field
context_data['Genr'] = context_data['Genr'].map(
    lambda x: ' '.join(
        stemmer.stem(word) for word in x.split()
        )
    )

df = df_cat_merge.merge(right = context_data,
                        left_on = 'pkgname',
                        right_on = 'ID')
# Remove duplicates and NAs
df = df.drop_duplicates()
df = df.dropna()
df = df.drop(columns='ID')
# Save file for later use
df.to_csv('CategoryStemming2Enhanced.txt', sep='\t', index=False)