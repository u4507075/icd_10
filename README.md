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
1. save_admit_data(): to read and decode the data including TXN, sex, age, weight, pluse rate, respiratory rate, body temperature, blood pressure, ABO blood group, Rh blood group, room that the patient admitted, and the last room when the patient was discharged, and ICD-10.
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
  
## Data analysis
Data from 2005 - 2016 are used to train machine learning models and data from 2017 are used to evaluate the models. We use overall accuracy, precision, recall, F-measure, and area under ROC curve to evaluate and compare predictive performance between models.

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
