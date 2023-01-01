# webpage_classifer
Classify web pages given the HTML text.

# Introduction

When we encounter a new website we need to know which category it belongs to, this would be helpful in webcrawling scnearios. We can use two type of modeling for this, 
- **Text models:**
    For text models we need to extarct the raw HTML and clean it and extract keywords and word frequencies and then use it for modeling. Here the issue would be we could have very big webpages and most of the data could be about website metadata and not what is rendered in the browser and we could misclassify a lot. Also the size of the text tokens amkes the use of transformers models difficult as most of the LLMs have a limit in input token size at 512.
- **Vision models:**
    To use vision models the websites that are rendered in the browser can be taken as screenshot and cropped to a predefined size and that can be used given to a classifer model or  zero shot classifictaion based on autoencoders



## Benchmark Text Model

**Dataset:** [Structured Web Data Extraction Dataset (SWDE)](https://academictorrents.com/details/411576c7e80787e4b40452360f5f24acba9b5159)

**Data split:** GroupShuffleSplit

**Features:** TfidfVectorizer

**Model:** MultinomialNB


Download the zip file and put it in the [data](./data) directory
Run the [data_process_and_train.ipynb notebook](notebooks\data_process_and_train.ipynb) to extarct the raw html and clean the text and prepare it for training

## Next steps

[Kaggle notebook](https://www.kaggle.com/code/abhishek/approaching-almost-any-nlp-problem-on-kaggle/notebook)

- Feature engineering
- More complex models
    - Random forest
    - XBBoost
    - LightGBM
- Hyperparamamter Tuning
    - n Fold CV
    - Grid / Random search

## Benchmark Vision Model

[Clustering websites with screenshots](https://sabrinas.space/)

**Dataset:** []()

**Data split:** 

**Features:** ResNet model

**Model:** KNN Clustering


