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
Over one million patient visit Maharaj Nakhon Chiang Mai hospital at the outer patient department (reported in [2018](http://www.med.cmu.ac.th/hospital/medrec/2011/index.php?option=com_content&view=category&id=130&Itemid=589)). Every year, the hospital needs to report data to the government for billing. 
### Problem statement
The amount of budget which can be claimed from the billing depends on the quality and completeness of the document. One major problem is the completeness of diagnosis (using [ICD-10](https://icd.who.int/browse10/2016/en), a classification standard for diagnosis). The current process to complete the diagnosis in the document is a labour intersive work which requires physicians or technical coders to review medical records and manually enter a proper diagnosis code. Therefore, we see a potential benefit of machine learning application to automate this ICD-10 labelling process.
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
1. The performance of machine learning model shows precision, recall, and F-measure greater than 80%.
2. Present one year cost-benefit analysis compared between before and after using machine learning models to fill ICD-10.

## Time line
### March 2019
  * Write and submit a research proposal and ethic.
  * Setup a new server.
  * Duplicate clinical data to the server.
  * Map and label column name and description.
  * Join the table data and create a single dataset.
### April 2019
  * Apply NLP and standard medical terminologies to preprocess input features.
  * Design and evaluate machine learning model.
### May 2019
  * Close the project either, the model performance is greater than 80% or it is the last week of May.
### June - August 2019
  * Write and submit a paper.
  
## Materials and methods
### Target group
Clinical records of outer-patient visits from 2006 - 2017 (2006 - 2016 for a training set, and 2017 for a test set) are retrospectively retrieved from the Maharaj Nakhon Chiang Mai electronic health records. Approximately one million records are expected to retrieve per year. Only encoded data (number, string) are included in the experiment (excluded images and scanned document).

### Dataset
Data recorded between 2006 - 2019 from the electronic health records of Maharaj Nakhon Chiang Mai were deidentified and preprocessed. All data that could be potentially able to track back to an individual patient such as patients' name, surname, address, national identification number, address, phone number, hospital number were removed. We used **TXN** (a unique number representing a patient visit) to be a joining key. The dataset was divided into five groups.

1. Registration data
2. Admission data
3. Laboratory data
4. Radiological report data
5. Drug prescription data

#### Registration data

The registration data is the demographic information of patients who visited (mostly outer patient department (OPD) cases) at Maharaj Nakhon Chiang Mai hospital. See the full detail of registration metadata [here](https://github.com/u4507075/icd_10/blob/master/REGISTRATION_METADATA.md).

#### Admission data

The admission data is the demographic information of patients who admitted to any internal wards (inner patient departments (IPD) cases) at Maharaj Nakhon Chiang Mai hospital. See the full detail of admission metadata [here](https://github.com/u4507075/icd_10/blob/master/ADMISSION_METADATA.md).

#### Laboratory data

See the full detail of laboratory metadata [here](https://github.com/u4507075/icd_10/blob/master/LAB_METADATA.md).

#### Radiological report data

The radiological report data is the reports that radiologists took notes after they reviewed the imaging. The notes were written in plain text describing the finding within the imaging and the impression of suspected abnormalities and/or provisional diagnosis. We **do not** include any image data in this experiment. The notes are required to preprocessed using natural language process techniques to clean and do feature engineering. This work is contributed in **radio** branch of this project.

#### Drug prescription data

The drug prescription data is the information of type of drugs which were prescribed to the patients. See the full detail of laboratory metadata [here](https://github.com/u4507075/icd_10/blob/master/DRUG_METADATA.md).

### Characteristics of dataset

**TXN** is a unique identification number of patient visit. TXN is a key to join across those five datasets (not always exists in all datasets). At the end of each visit, the diagnoses (ICD-10) relating to the patient had to enterred to the database. You will have to build an approach and develop machine learning models to extract patterns which are able to correctly enter ICD-10. However, you will face some problems with the datasets.

1. Unidentified specific TXN to ICD-10: We do not know that what prescriptions, laboratory findings, and radiological reports relate to which ICD-10. For example, in one visit (one TXN), a patient might have 3 diagnoses and get prescription with 10 drugs. We do not know which drug is prescribed for which diagnosis.

2. Large number of ICD-10 (target class): The total number of ICD-10 is 38,970 types (always starts with A-Z) and approximately 14,000 of them were used in the records.   

### Data preprocessing
All identification data such as name, surname, address, national identification, hospital number will be removed according to patient privacy. Data from the five resources were pre-processed and transformed to numeric type and stored in several datasets. TXN was used as a key to link between datasets. Each dataset contains TXN, input features, and ICD-10 (as a target label. All data were decoded to **tis-620** in order to read Thai language. Data in Thai were not used as input features but could be used for human validation step.

#### Registration data
1. save_admit_data(): Read and decode the data including TXN, sex, age, weight, pluse rate, respiratory rate, body temperature, blood pressure, ABO blood group, Rh blood group, room that the patient admitted, and the last room when the patient was discharged, and ICD-10.

| Features | Mapping criteria |
| :--- | :--- |
| TXN | numeric |
| sex | 'm','f' |
| age | numeric |
| weight | numeric |
| pulse rate | numeric |
| respiratory rate | numeric |
| body temperature | numeric |
| blood pressure | split text by '/' to systolic and diastolic blood pressure |
| ABO blood group | 'a','b','ab','o' |
| Rh blood group | 'p','n' |
| Room | to lowercase, remove * and space |
| ICD-10 | string |

2. onehot_admit_data(): Apply onehot encoding to sex, ABO blood group, Rh blood group, and room. Then, convert all values to numeric.  

#### Admission data
Apply the same process to registration data.

#### Laboratory data
1. get_lab_data(config): Read and decode the data including TXN, lab code, and ICD-10. Select only lab code that the number is larger than 500 records.
2. split_lab_data(): Laboratory results were stored in text separated by ';'. This process split the text by ';' to obtain a single value of laboratory result.
3. clean_lab_data(): remove all symbols and space, group words indicating negative results to negative and words indicating positive results to positive, and remove all English alphabet from number.
4. get_encode_lab(): Get a unique list of clean text results and map with unique numbers.
5. encode_lab_data(): Use the encoding map to transform text result to number.

#### Radiological report data
 

#### Drug prescription data
1. getdata(config,'table_name','drug): Read and decode the data including TXN, drug code, and ICD-10 where 'table_name' is the table of opd and ipd cases.
2. remove_space_data('drug'): Remove space from drug code.
3. get_encode_feature('drug'): Get a unique list of clean text results and map with unique numbers.
4. encode_feature('drug'): Use the encoding map to transform text result to number.

## Split training and test set and evaluation metrics
Data from January 2005 - April 2017 are used to train machine learning models and data after April 2017 are used as a test set to evaluate the models. We use overall accuracy, precision, recall, F-measure, and area under ROC curve to evaluate and compare predictive performance between models.

Dataset (training and test set)

| Dataset | csv files |
| :--- | :--- |
| registration data | registration_onehot.csv |
| admission data | admit_onehot.csv |
| laboratory data | B06.csv,B13.1.csv,B13.csv,GMCL001.csv,L01.csv,L022.csv,L024.csv,L025.csv,L0261.csv,L027.csv,L029.csv,L02.csv,L0371.csv,L037.csv,L0414.csv,L0421.csv,L071.csv,L073.csv,L078.csv,L07.csv,L083.csv,L090.csv,L091.csv,L093.csv,L1001.csv,L10041.csv,L10044.csv,L1005.csv,L101763.csv,L1022.csv,L1030.csv,L1031.csv,L1032.csv,L1040.csv,L10501.csv,L10502.csv,L1052.csv,L10561.csv,L105621.csv,L1056221.csv,L10573.csv,L10591.csv,L105932.csv,L105933.csv,L106011.csv,L107011.csv,L107018.csv,L1081.csv,L1084.csv,L10962.csv,L10981.csv,L1901.csv,L1902.csv,L1903.csv,L1904.csv,L1905.csv,L1906.csv,L1907.csv,L1910.csv,L1911.csv,L1914.csv,L202.csv,L2082.csv,L36.csv,L421.csv,L422.csv,L4301.csv,L531.csv,L54.csv,L551.csv,L5712.csv,L581.csv,L58.csv,L61.csv,L84.csv |
| radiological report data | numeric |
| drug prescription datat | drug_numeric.csv |

## Design the approach to predict ICD-10
### Approach 1: Multi-class classification
We simply train machine learning models with all ICD-10 as a target class and evaluation result showed that the model accuracy was less than 0.01 %. We discuss that the poor performance of the models causes by the large number of ICD-10 and unidentified specific TXN to ICD-10.

### Approach 2: Binary classification
Because of those two problems, we change the approach from multi-class to binary classification. We select one ICD-10 at a time as a target class and randomly select the same number of instances labelled with target and non-target class as training and test, respectively. Then, train Xgboost (wiht max-depth 100) to create a model per ICD-10. The binary classification approach helps to reduce the complexity of the model (approaching a target class one by one instead of all classes in the same time) and provide flexibility to predict more than one ICD-10 per instance. The model evaluation (separate model) showed that the accuracy ranges between 50 - 100 % (average 60%).

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
