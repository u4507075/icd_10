# Auto-mapping ICD-10 using Radiological reports

## Problems
We know that radiological reports contain a lot of useful information. In the reports, radiologist describes findings and provides suggestions for a provisional diagnosis which could be related to the final diagnosis of patients. We see this potential pattern can help us to map ICD-10 (diagnosis) based on the knowledge derived from the radiological reports. However, the reports are in plain text which is pretty hard to extract information. We set up this sandbox git branch **"radio"** to play around how we could extract useful information from the reports and use the information to map a correct ICD-10

## Plan
1. Use NLP techniques to preprocess and extract relevant features from the reports.
2. Use the features to train machine learning models.
3. Evaluate and discuss the model performance.
4. Integrate the best (accuracy and clinical validity) to the main project (where multiple models use different dataset to predict ICD-10).

## Data type
1. Type of imaging: plain file, CT, MRI, etc (text).
2. Location of investigatin: head, chest, abdoment, etc (text).
3. Radiological findings: describe relevant (normal/abnormal) findings of the investigation results, suggest diagnosis or abnormalities related to the patient based on the evidences.

## Review matching patterns between the reports and ICD-10
1. Exact match
2. xxx
3. xxx
