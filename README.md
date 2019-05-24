# Auto-mapping ICD-10 using machine learning model

## Researchers
1. Assistant Professor Piyapong Khumrin, MD, Faculty of Medicine, Chiang Mai University, Chiang Mai, Thailand
2. Assistant Professor Krit Khwanngern, MD, Faculty of Medicine, Chiang Mai University, Chiang Mai, Thailand
3. Associate Professor Nipon Theera-Umpon, PhD, Biomedical Engineering Institute, Chiang Mai University
4. Terence Siganakis, CEO, Growing Data Pty Ltd
5. Alexander Dokumentov, Data Scientist, Growing Data Pty Ltd

## Technical support
1. Atcharaporn Angsuratanawech 
2. Sittipong Moraray
3. Pimpaka Chuamsakul
4. Pitupoom Chumpoo
5. Prawinee Mokmoongmuang

## Duration
6 months (March - August 2019)

## Introduction 
Annually, over one million patients visit the out-patient departments of the Maharaj Nakhon Chiang Mai hospital (reported in [2018](http://www.med.cmu.ac.th/hospital/medrec/2011/index.php?option=com_content&view=category&id=130&Itemid=589)). Every year, the hospital needs to report data to the government for billing.

### Problem statement
The amount of the budget which can be claimed from the billing depends on the quality and completeness of the documentation. One major problem is the completeness of diagnosis (using [ICD-10](https://icd.who.int/browse10/2016/en), a classification standard for diagnosis). The current process to complete the diagnosis in the documents is a labor intensive work which requires physicians or technical coders to review medical records and manually enter a proper diagnosis code. Therefore, we see a potential benefit of machine learning application to automate this ICD-10 labeling process.

### Prior work
[ICD-10](https://en.wikipedia.org/wiki/ICD-10) is a medical classification list for medical related terms such as diseases, signs and symptoms, abnormal findings, defined by the World Health Organization (WHO). In this case, ICD-10 is used to standardized the diagnosis in the billing report before submitting the report to the government. Prior research showed the success of applying machine learning for auto-mapping ICD-10. 

[Serguei et al.](https://academic.oup.com/jamia/article/13/5/516/734238) applied simple filters or criteria to predict ICD-10 code such as assuming that a high frequency code that shares the same keywords is a correct code, or gender specific diagnsis (use gender to separate female and male specific diseases). Other cases which were not solved by those filters, then use a machine learning model (Naive Bayes) or bag of words technique. They used the SNoW implementation of the naïve Bayes classifier to solve the large number of classification, and Phrase Chunker Component for mixed approach to solve a classification. The model evaluation showed that over 80% of entities were correctly classified (with precision, recall, and F-measure above 90%). The limitations of the study were the features were treated independently which might lead to a wrong classification, continuous value such as age interfered how the diseases are classified, and lower number of reference standard (cased manually labelled by human coder).

[Koopman et al.](https://www.sciencedirect.com/science/article/pii/S1386505615300289) developed a machine learning model to automatically classify ICD-10 of cancers from free-text death certificates. Natural language processing and SNOMED-CT were used to extract features to term-based and concept-based features. SVM was trained and deployed into two levels: 1) cancer/nocancer (F-measure 0.94) and 2) if cancer, then classify type of cancer (F-measure 0.7). 

[Medori and Fairon](https://aclanthology.info/pdf/W/W10/W10-1113.pdf) mapped clinical text with standardized medical terminologies (UMLS) to formulate features to train a Naive Bayes model to predict ICD-6(81% recall). 

[Boytcheva](http://www.aclweb.org/anthology/W11-4203) matched ICD-10 codes to diagnoses extracted from discharge letters using SVM. The precision, recall, F-measure of the model were 97.3% 74.68% 84.5%, respectively. 

In summary, prior research shows that machine learning model plays a significant and beneficial role in auto-mapping ICD-10 to clinical data. The common approach of the preprocessing process is using NLP process to digest raw text and map with standardized medical terminologies to build input features. This is the first step challenge of our research to develop a preprocessing protocol. Then, the second step is to design an approach how to deal with a large number of input features and target classes (ICD-10).

Our objectives are to develop machine learning model to mapp missing or verify ICD-10 in order to obtain more complete billing document. We aim to test if we use the model to complete ICD-10 for one year report and evaluate how much more the hospital can claim the billing. 

## Objectives
1. Use machine learning models to predict missing ICD-10.
2. Use machine learning models to verify ICD-10 labelled by human.

## Aims
1. The performance of machine learning model (accuracy, precision, recall, and f-measure) is greater than 80%.
2. Present one year cost-benefit analysis compared between before and after using machine learning models to fill ICD-10.

## Time line
### March - April 2019
  * Write and submit a research proposal and ethic.
  * Setup a new server.
  * Duplicate clinical data to the server.
  * Map and label column name and description.
  * Join the table data and create a single dataset.
### May - June 2019
  * Apply NLP and standard medical terminologies to preprocess input features.
  * Develop and evaluate machine learning model.
### June 2019
  * Close the project either, the model performance is greater than 80% or it is the end of June.
### July - August 2019
  * Write and submit a paper.
  
## Materials and methods
### Target group
Clinical records of out-patient visits from 2006 - 2017 (2006 - 2016 for a training set, and 2017 for a test set) are retrospectively retrieved from the Maharaj Nakhon Chiang Mai electronic health records according to the approval of ethical exemption. Approximately one million records are expected to be retrieved per year. Only encoded data (number, string) are included in the experiment (excluding images and scanned documents).

### Dataset
Data recorded between 2006 - 2019 from the electronic health records of Maharaj Nakhon Chiang Mai were de-identified and preprocessed. All data that could potentially be used to track back to an individual patient such as patients' name, surname, address, national identification number, address, phone number, hospital number were removed. We use **TXN** (a unique number representing a patient visit) to be a joining key between five groups of datasets (total size 82G).

#### Raw dataset
1. Registration data (reg.csv (176M))
2. Admission data (adm.csv (147M))
3. Laboratory data (lab.csv (3.1G), ilab.csv(74G))
4. Radiological report data (rad.csv (107M), irad (107M))
5. Drug prescription data (dru.csv (189M), idru (3.9G))

TXN is a joining key for reg.csv, lab.csv, rad.csv, and dru.csv.
TXN is a joining key for adm.csv, ilab.csv, irad.csv, and idru.csv.
**DO NOT** use TXN to join across groups.
The file "icd10.csv" (9.5M) is a for mapping ICD-10 code (using in the datasets) to ICD-10 description.

#### Registration data

The registration data is the demographic information of patients who visited at the outer patient department (OPD) at Maharaj Nakhon Chiang Mai hospital. 

| Features | Types | Description |
| :--- | :--- | :--- |
| txn | numeric | key identification for a patient visit |
| sex | categorical | m = male, f = female |
| age | numeric | age (year) |
| wt | numeric | weight (kg) |
| pulse | numeric | pulse rate (times/min) |
| resp | numeric | respiratory rate (times/min) |
| temp | numeric | body temperature (celcius) |
| sbp | numeric | systolic blood pressure (mmHg) |
| dbp | numeric | diastolic blood pressure (mmHg) |
| blood | categorical | a = blood group A b = blood group B, o = blood group O, ab = blood group AB 
| rh | categorical | n = blood group Rh negative, p = blood group Rh positive |
| room | string | Room codes that patients visited |
| icd10 | string | ICD-10 code (diagnosis) |

#### Admission data

The admission data is the demographic information of patients who admitted to the internal wards (inner patient departments (IPD) cases) at Maharaj Nakhon Chiang Mai hospital. 

| Features | Types | Description |
| :--- | :--- | :--- |
| txn | numeric | key identification for a patient visit |
| sex | categorical | m = male, f = female |
| age | numeric | age (year) |
| wt | numeric | weight (kg) |
| pulse | numeric | pulse rate (times/min) |
| resp | numeric | respiratory rate (times/min) |
| temp | numeric | body temperature (celcius) |
| sbp | numeric | systolic blood pressure (mmHg) |
| dbp | numeric | diastolic blood pressure (mmHg) |
| blood | categorical | a = blood group A b = blood group B, o = blood group O, ab = blood group AB 
| rh | categorical | n = blood group Rh negative, p = blood group Rh positive |
| room | string | Room codes that patients admitted |
| room_dc | string | Room codes that patients discharged |
| icd10 | string | ICD-10 code (diagnosis) |

#### Laboratory data
The laboratory data is Laboratory findings investigated in OPD (lab.csv) and IPD (ilab.csv) cases.

| Features | Types | Description |
| :--- | :--- | :--- |
| txn | numeric | key identification for a patient visit |
| lab_name | string | Lab code |
| name | string | Lab items within the lab code |
| value | object | value of lab items (can be numeric (only value) or string (value with unit)) |
| icd10 | string | ICD-10 code (diagnosis) |

#### Radiological report data

The radiological report data is the reports that radiologists took notes after they reviewed the imaging. The notes were written in plain text describing the finding within the imaging and the impression of suspected abnormalities and/or provisional diagnosis. We **do not** include any image data in this experiment. The notes are required to preprocessed using natural language process techniques to clean and do feature engineering. Radiological report of OPD and IPD cases were stored in the rad.csv and irad.csv, respectively.

| Features | Types | Description |
| :--- | :--- | :--- |
| txn | numeric | key identification for a patient visit |
| location | string | location of examination such as cest, hand, abdomen |
| position | string | position of examination such as plain film, posteroanterior (pa), lateral |
| report | string | radiological report |
| icd10 | string | ICD-10 code (diagnosis) |

#### Drug prescription data

The drug prescription data is the information of type of drugs which were prescribed to the patients. 

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| drug | string | Drug code |
| drug_name | string | Drug name with or without description |
| icd10 | string | ICD-10 code (diagnosis) |

### Limitations of the dataset

**TXN** is a unique identification number of patient visit. TXN is a key to join across those opd and ipd datasets (not always exists in all datasets). At the end of visit, the diagnoses (ICD-10) relating to the patient were enterred to the database which linked to one TXN. 

1. No individual information mapped to specific ICD-10: As one TXN can contain several ICD-10, we do not know which prescriptions, laboratory findings, and radiological reports relate to which ICD-10. For example, in one visit (one TXN), a patient might have 3 diagnoses and get prescription with 10 drugs. We do not know which drug is prescribed for which diagnosis.

2. Large number of ICD-10 (target class): The total number of ICD-10 is 38,970 types (always starts with A-Z) and approximately 14,000 of them were used in the records.   

### Data preprocessing
All identification data such as name, surname, address, national identification, hospital number will be removed according to patient privacy. Data from the five resources were pre-processed and transformed to numeric type and stored in several datasets. TXN was used as a key to link between datasets. Each dataset contains TXN, input features, and ICD-10 (as a target label. All data were decoded to **tis-620** in order to read Thai language. Data in Thai were not used as input features but could be used for human validation step. Data were preprocessed (see more details in [copy.py](https://github.com/u4507075/icd_10/blob/dev/src/preprocessing/text.py)) including removing space, stop words, outliers, filterring relevant data, embedding words. Then use [spacy](https://spacy.io/) to transform words to vector. 

## Split training and test set and evaluation metrics
Data from January 2005 - April 2017 are used to train machine learning models and data after April 2017 are used as a test set to evaluate the models. We use overall accuracy, precision, recall, F-measure, and area under ROC curve to evaluate and compare predictive performance between models.

## Scale dataset
Because machine learning algorithms work better when features are on a relatively similar scale and close to normally distributed ([refer to this article](https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02), we applied two keras preprocessing scalers (MinMaxScaler and StandardScaler) to normalise the dataset. We could not use other scaler methods because those methods requires to read the entire datset at the same time whereas MinMaxScaler and StandardScaler allow us to do partial fit. MinMaxScaler scale raw data to preferred range (in this case between 0 - 1) and still preserve the same distribution of the dataset. Thus, the scaler does not reduce the effect of outliers but it helps to scale all features to be in the same proprotion range. 

### MinMaxScaler
```
new_value = (x - min(feature))/(max(feature)-min(feature))
```
StandardScaler standardise dataset using mean and standard deviation of the feature. The result of StandardScaler scale changes the distribution of data to normal with a standard deviation equal to 1. Thus, StandardScaler changes the distribution of the dataset and reduces the effect of outliers but the data range is bigger than MinMaxScaler. Deep learning learn better when the data is normal distributed. So, we more likely to choose StandardScaler to normalise the data when we approach with LSTM.

### StandardScaler
```
new_value = (x - avg(feature))/(sdev(feature))
```

## Design the approach to predict ICD-10
### Approach 1: Multi-class classification
#### Scale dataset
We use spacy to transform entire string data to numeric data, scale the input features using StandardScaler encode ICD-10, and map ICD-10 to number.
#### Incremental models
We use the models which support partial fit because the training set is too big to fit in one time. The models include multiclass classifiers (PassiveAggressiveClassifier, SGDClassifier, and Perceptron) and regression classifiers (PassiveAggressiveRegressor and SGDRegressor) using [dask-ml](https://examples.dask.org/machine-learning/incremental.html#Model-training) library. The models are trained with the training set and saved as .pkl file. Then, load the models to validate with the testset.
#### LSTM
We feed data to train LSTM (3 layers 512x512x512) with all ICD-10 as a target class and initially evaluation the training loss again evaluation loss. The loss shows that .......(still training the model).

We test the model with a test instance by:
1. Enter the input values to all ICD-10 model. 
2. Return positive result if the probability of model prediction of the target class is larger than 0.9.
3. Store the positive ICD-10 result in a list sorted by probability deascendingly.
4. Validate the accuracy by counting how many actual ICD-10 matches with the predicted ICD-10.

## How to use
1. Clone the project and change to dev branch
```
git clone https://github.com/u4507075/icd_10.git
cd icd_10
git checkout dev
```
2. Check out and update dev branch
```
git fetch
git checkout dev
git pull
```
3. Commit and push
```
git add .
git commit -m "your message"
git push

#check remote
git remote -v
```
## How it works
## Model evaluation
## Limitations
