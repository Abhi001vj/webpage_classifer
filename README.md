# webpage_classifer
Classify web pages given the HTML text.


## Benchmark model

Dataset: [Structured Web Data Extraction Dataset (SWDE)](https://academictorrents.com/details/411576c7e80787e4b40452360f5f24acba9b5159)
Data split: GroupShuffleSplit
Features: TfidfVectorizer
Model: MultinomialNB

Downlaod the zip file and put it in the [data](./data) directory
Run the [data_process_and_train.ipynb notebook](notebooks\data_process_and_train.ipynb) to extarct the raw html and clean the text and prepare it for training

## Next steps

[Kaggle notebook](https://www.kaggle.com/code/abhishek/approaching-almost-any-nlp-problem-on-kaggle/notebook)
- Feature engineering
    - 
- More complex models
    - Random forest
    - XBBoost
    - LightGBM
- Hyperparamamter Tuning
    - n Fold CV
    - Grid / Random search


