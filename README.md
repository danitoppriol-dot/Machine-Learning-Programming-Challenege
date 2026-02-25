# Machine-Learning-Programming-Challenege
Programming Challenge of a course that I did

# Assignment description from professors

You must build and train a classifier given a labeled dataset and then use it to infer the labels of a given unlabeled evaluation dataset. You must then submit the inferred labels in a specified format, which will be compared to our evaluation data that has the correct labels (not shared, held out). If you want an unbiased estimate of how your model is performing, you can do your own train/test split using the supplied training data. After the competition/exam we will also publish the held-out true labels enabling you to compute your exact score.

The accuracy achieved by your model on our (held out) evaluation data maps to the score that it will receive in the challenge according to the curve below:
<img width="841" height="558" alt="image" src="https://github.com/user-attachments/assets/80179b6f-2cb7-479c-872b-d71746066418" />

Some logistics:

This challenge is to be done individually. The work you submit must be your own.
You can use whatever programming language and libraries you want.  You can use code you wrote for the labs. The challenge is designed such that it does not require high computational resources, but you can use Google Colab (Links to an external site.) if you feel the need.
The training and evaluation dataset files are formatted as comma-separated values, with each line being an observation. Like real data, there may be problems with some of the entries in the training dataset file.
You must submit two things: 1) your code (a zipfile is fine, but NO OTHER compression, e.g., rar); 2) a text file (with ".txt" as an extension) where each line is ONLY the label inferred by your system in the same order as that of the evaluation dataset file. If you do not submit these TWO files in these ways you will receive ZERO points. If you send for example a rar compressed file you will also receive a ZERO.
If you submit only one zip file containing your code and predicted labels, you will receive ZERO.
Be sure the labels your system generates are the same as those appearing in the training dataset. For instance, if the labels in the training dataset are {"andre", "atsuto", and "jörg"}, and the labels you predict are {"andre", "atsuto", and "jorg"}, all your predictions of "jorg" will be incorrect.
Your label file should not contain any extra data such as a header or index column. If your label file has a header or index column, you will receive ZERO.
Make sure the classifier you use in the end is the best you think you can create.
No questions will be answered from the instructors. Ask the data!
Use this opportunity to prepare for the exam!
 

The full data, including groundtruth, is now published below:

Here's the training data  (variables x and example labels y): TrainOnMe_orig.csvDownload TrainOnMe_orig.csv

Here's the evaluation data (only variables x, you have to infer labels y): EvaluateOnMe.csvDownload EvaluateOnMe.csv


NEW: Here is the ground truth data set (correct labels): EvaluationGT.csv
