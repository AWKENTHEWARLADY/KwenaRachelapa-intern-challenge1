# Prompt Log
This log shows how AI assisted during debugging of a machine learning pipeline.
## Prompt 1
help me identify data leakage in the ml pipeline and how to solve it

## Outcome
I learned that data leakage happens when the model uses information that would not exist during real prediction.  
In this project, the complaint_id prefix revealed the target class.  
I removed id_prefix and any features derived from complaint_id to fix this issue.



## Prompt 2
explain why is it a problem when it comes to fitting tf-idf before train/test split

## Outcome
I learned that fitting TF-IDF before splitting causes the model to learn vocabulary and word statistics from the test set.  
This leaks information into training and gives unrealistic performance.  
I fixed it by splitting the dataset first and fitting TF-IDF only on training data.



## Prompt 3
how does class imbalance affects accuracy

## Outcome
I learned that accuracy becomes misleading when one class dominates the dataset.  
The model can predict the majority class often and still achieve high accuracy.  
I added precision, recall, and F1-score to evaluate performance properly.



## Prompt 4
report label mapping

## Outcome
I learned that LabelEncoder assigns numeric values based on alphabetical order.  
This can cause mismatch between predicted labels and displayed class names if manually set.


## Prompt 5
How do I fix incorrect label mapping in classification_report?

## Outcome
I fixed the issue by using:
- label_encoder.classes_

This ensures the classification report uses the correct class order automatically instead of hardcoded labels.



## Prompt 6
What is the difference between fit_transform and transform?

## Outcome
I learned:
- fit_transform learns patterns from data and applies transformation
- transform only applies an already learned transformation

I used:
- fit_transform on training data only
- transform on test data to avoid data leakage


#Here is the link for my chats
https://chatgpt.com/share/6a028ca3-7064-83ea-af91-cb347af349c2

# Summary
This debugging process improved my understanding of
data leakage prevention
correct machine learning pipeline structure
evaluation metrics beyond accuracy
correct label encoding usage
- proper TF-IDF usage in NLP pipelines