## Math-Preprint-Classifier

End-to-end model for predicting the [MSC (Mathematical Subject Classification)](https://en.wikipedia.org/wiki/Mathematics_Subject_Classification) codes (as of [2020](https://mathscinet.ams.org/mathscinet/msc/msc2020.html)) and the primary [arXiv category](https://arxiv.org/category_taxonomy) for a preprint from a math archive or from the mathematical physics archive.

The project involved: 
1) scraping preprints from arXiv and preprocessing the data from scratch, 
2) exploratory data analysis, 
3) training models on the text data (titles and abstracts) for predicting 2 and 3-character MSC classes (multi-label classification) as well as for predicting the math archive primary category (multi-class classification),
4) building a pipeline for an end-to-end model which internally takes care of fetching the data, text cleaning and different prediction tasks. 

### Usage
The end-to-end classifier is an instance of the `math_classifier` class coded in `end_to_end.py`.

![alt text](https://github.com/FilomKhash/Math-Preprint-Classifier/blob/main/image1.png)

Upon instantiation, saved models for individual tasks (predicting 2 or 3-character MSC classes or predicting the primary category) are loaded from the `models` folder along with the names of their variables from the `data` folder. The `predict` method can receive raw text data, i.e. the title and/or the abstract of a math-related paper, an [arXiv identifier](https://info.arxiv.org/help/arxiv_identifier.html), or a number for how many papers to scrape. It outputs the predicted MSC classes and the primary category along with their probabilities.

![alt text](https://github.com/FilomKhash/Math-Preprint-Classifier/blob/main/image2.png)

### Examples

Check notebook `example_jupyter.ipynb`. Alternatively, `example_cmd.py` can be run from the terminal: five recent math-related submissions from arXiv are fetched and their URL's are printed along with the predicted labels and their probabilities.

### Data Collection and Cleaning

The notebook `Scarping and Cleaning the Data.ipynb` involves a step-by-step process in which numerous arXiv preprints associated with at least one MSC class are gathered; their primary category and MSC classes are recorded for multi-class and multi-label classification tasks respectively; and their text data (title+abstract) is cleaned (removing punctuations, special characters, stop words, math environment etc.). The final dataset has more than 160,000 entries and can be found in the folder `data`.  

### Trained Models

Based on the scraped data, in `MSC Prediction.ipynb` we train a support vector machine for the following multi-label text classification task:

$\hspace{3cm}$  `cleaned_text` $\mapsto$ 3-character MSC classes (e.g. 46L, 57M, 57R) $\hspace{1cm}$  (~500 labels, trained on ~100,000 data points).

The coarser target variable, 2-character MSC classes, can then be predicted too:

$\hspace{3cm}$  `cleaned_text` $\mapsto$ 3-character MSC classes (e.g. 46L, 57M, 57R) 
$\mapsto$ 2-character MSC classes (e.g. 46, 57) $\hspace{1cm}$  (~60 labels).

As for the primary arXiv category of a math-related paper, in `Math Archive Primary Category Prediction.ipynb` we train a convolutional neural net for this multi-class task:

$\hspace{3cm}$  `cleaned_text`  $\mapsto$ the primary arXiv category (e.g. `math.AG`) $\hspace{1cm}$  (~30 labels, trained on ~120,000 data points).

The performances over the test sets are as follows<a name="cite_ref-a"></a>[<sup>*</sup>](#cite_note-a):

|    Target Variable(s)                  | Weighted $F_1$ Score |  Jaccard Score/Accuracy         |
| -------------------------------------- | -------------------- | ------------------------------- |
|3-character MSC classes (~500 labels)   | 50.21%               | 39.03%                          |
|2-character MSC classes (~60 labels)    | 65.1%                | 56.32%                          |
|the primary arXiv category (~30 labels) | 65.80%               | 65.52%                          |


<a name="cite_note-a"></a>[*](#cite_ref-a) The metrics were computed on test sets with ~40,000 data points, and all are recorded as percentages. On the first two rows, the task is multi-label and the average of the Jaccard similarity over test instances is used while on the last row the accuracy is recorded for the multi-class task.   
