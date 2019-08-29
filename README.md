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
### 31 August 2019
  * Model validation (first round).
### 30 September 2019
  * Model validation (second round).
### 31 October 2019
  * Model validation (third round).
### November 2019
  * Discuss the result
### December
  * Close the project and write a manuscript.

  
## Materials and methods
### Target group
Clinical records of out-patient visits from 2006 - 2017 (2006 - 2016 for a training set, and 2017 for a test set) are retrospectively retrieved from the Maharaj Nakhon Chiang Mai electronic health records according to the approval of ethical exemption. Approximately one million records are expected to be retrieved per year. Only encoded data (number, string) are included in the experiment (excluding images and scanned documents).

### Dataset
Data recorded between 2006 - 2019 from the electronic health records of Maharaj Nakhon Chiang Mai were de-identified and preprocessed. All data that could potentially be used to track back to an individual patient such as patients' name, surname, address, national identification number, address, phone number, hospital number were removed. We use **TXN** (a unique number representing a patient visit) to be a joining key between five groups of datasets (total size 82G). Raw dataset is the dataset that contain orignal data (numeric and string format). Vec dataset is the dataset that were converted to numeric format using wordtovec.

#### Raw dataset
1. Registration data (reg.csv (161M))
2. Admission data (adm.csv (133M))
3. Laboratory data (lab.csv (2.9G), ilab.csv(67G))
4. Radiological report data (rad.csv (99M), irad (653M))
5. Drug prescription data (dru.csv (172M), idru (3.5G))

#### Vec dataset
1. Registration data (reg.csv (378M))
2. Admission data (adm.csv (324M))
3. Laboratory data (lab.csv (2.6G), ilab.csv(57G))
4. Radiological report data (rad.csv (2.3G), irad (15G))
5. Drug prescription data (dru.csv (169M), idru (3.2G))

TXN is a joining key for reg.csv, lab.csv, rad.csv, and dru.csv.
TXN is a joining key for adm.csv, ilab.csv, irad.csv, and idru.csv.
**DO NOT** use TXN to join across groups.
The file "icd10.csv" (9.5M) is a for mapping ICD-10 code (using in the datasets) to ICD-10 description.

#### Registration and admission data

The registration data is the demographic information of patients who visited at the outer patient department (OPD) at Maharaj Nakhon Chiang Mai hospital. The admission data is the demographic information of patients who admitted to the internal wards (inner patient departments (IPD) cases) at Maharaj Nakhon Chiang Mai hospital. The structure of both dataset is the same except there is no room_dc (assign zero to all instances) in registration data.

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
| room_dc | string | Room codes that patients discharged (available only in admission dataset)|
| dx_type | numeric | 0 = missing data, 1 = principal diagnosis, 2 = co-morbidity, 3 = complication, 4 = other, 5 = external injury |
| icd10 | string | ICD-10 code (diagnosis) |

#### Laboratory data
The laboratory data is Laboratory findings investigated in OPD (lab.csv) and IPD (ilab.csv) cases.

| Features | Types | Description |
| :--- | :--- | :--- |
| txn | numeric | key identification for a patient visit |
| lab_name | string | Lab code |
| name | string | Lab items within the lab code |
| value | object | value of lab items (can be numeric (only value) or string (value with unit)) |
| dx_type | numeric | 0 = missing data, 1 = principal diagnosis, 2 = co-morbidity, 3 = complication, 4 = other, 5 = external injury |
| icd10 | string | ICD-10 code (diagnosis) |

#### Radiological report data

The radiological report data is the reports that radiologists took notes after they reviewed the imaging. The notes were written in plain text describing the finding within the imaging and the impression of suspected abnormalities and/or provisional diagnosis. We **do not** include any image data in this experiment. The notes are required to preprocessed using natural language process techniques to clean and do feature engineering. Radiological report of OPD and IPD cases were stored in the rad.csv and irad.csv, respectively.

| Features | Types | Description |
| :--- | :--- | :--- |
| txn | numeric | key identification for a patient visit |
| location | string | location of examination such as cest, hand, abdomen |
| position | string | position of examination such as plain film, posteroanterior (pa), lateral |
| report | string | radiological report |
| dx_type | numeric | 0 = missing data, 1 = principal diagnosis, 2 = co-morbidity, 3 = complication, 4 = other, 5 = external injury |
| icd10 | string | ICD-10 code (diagnosis) |

#### Drug prescription data

The drug prescription data is the information of type of drugs which were prescribed to the patients. 

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| drug | string | Drug code |
| drug_name | string | Drug name with or without description |
| dx_type | numeric | 0 = missing data, 1 = principal diagnosis, 2 = co-morbidity, 3 = complication, 4 = other, 5 = external injury |
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
### Target class

The main problem is the number of target class (icd10) is large (14,000). Moreover, not all input correctly associates to a target class. The target class could be represented in 4 ways.
1. Individual class: one icd10 is treated as one target class. This approach creates the complexity and redundancy of data which makes the models hard to learn.
2. Group class: group all icd10s in one txn and embed to a new target class. The combination of icd10s is found that often repeated because the diseases usually relate to each other. For instance, acute renal failure is commonly found with sepsis because sepsis leads to shock and shock causes renal failure.

###SNOMED CT
[SNOMED CT](http://this.or.th/files/44_SNOMED_CT_Manual_VersionThai_201903.pdf?fbclid=IwAR3u0cZyXwefnVH2uCzwqHPC_kHs6ZR1ef6DtDe8yA60R6JkiCbfC53wZP4) is a systematically organized collection of medical terms. SNOMED CT maps medical terms or phrases to a concept id which represents the association to computer processable hirachical concept. SNOMED CT also maps concept id to icd10 which we could use these association to map text from radiological report to icd10 through concept id. 

## Design the approach to predict ICD-10
### Approach 1: Multi-class classification
#### Scale dataset
We use spacy to transform entire string data to numeric data, scale the input features using StandardScaler encode ICD-10, and map ICD-10 to number.
#### Incremental models
We use the models which support partial fit because the training set is too big to fit in one time. The models include multiclass classifiers (PassiveAggressiveClassifier, SGDClassifier, and Perceptron) and regression classifiers (PassiveAggressiveRegressor and SGDRegressor) using [dask-ml](https://examples.dask.org/machine-learning/incremental.html#Model-training) library. The models are trained with the training set and saved as .pkl file. Then, load the models to validate with the testset.
#### Clustering
[Birch clustering algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch)

As we know, there are too large number of target classes (icd10) which the common supervised ml model might not be very fit to solve the problem. Also, the data do not provide a clear cut where which drug, investigation pairs with a single diagnosis.

Thus, clustering might be a good option to start with because it groups similar things together without the need that you have to separate the instance into a single grain level.

There are so many kinds of clustering algorithms. Which one should I choose?

Have a look at the comparison of the clustering algorithms in [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering)

The preferred algorithm should be able to:

1. Handle very large n-samples (can do partial_fit)
2. Good with large n-cluster (handle large number of icd10)
3. No need to define n-cluster before training

From the list of clustering algorithms, [Birch](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch) and [MiniBatchKmeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans) fullfil the criteria. In Birch, we can either set threshold (how far of neighbour instances that should be separated as a new cluster) or number of cluster where as MiniBatchKmeans could be only set the number of cluster. Low threshold means instances within the same cluster are very close to each other.

Next, we have to choose an optimal number of cluster. We know that the higher number of cluster creates smaller clusters and usually lower error rate but trends to overfitting. There are several [ways](https://jtemporal.com/kmeans-and-elbow-method/) to decide depending on your objectives. In this case, we aim to get each cluster as large as possible to represent one ICD-10. Approximately 15,000 ICD-10s were recorded in the database. We use this number to determine the target number of cluster. We can directly set the number of cluster in MiniBatchKMeans whereas we need to fine tune the threshold to get the number of cluster as close as 15,000 in Birch.

| Dataset | MiniBatchKmeans (n_cluster) | Birch (threshold,n_cluster) |
| :--- | :--- | :--- |
| reg and adm | 15,000 | (3.0117,14983) |
| lab and ilab | 15,000 | xxx |
| dru and idru | 15,000 | (0.1375,15010) |
| rad and irad | 15,000 | xxx |

The models were trained with the training dataset (dru and idru csv files) then use the trained models predict cluster using the same training dataset. Then, aggregate the number of icd10 in the cluster. As mentioned, drug with highly specific to a particular diagnosis presents a strong pattern that the ratio of number of icd10 inside and outside cluster is high, defined as weight.

```
weight = number of icd10(x) in the cluster/total number of icd10(x)
```

However, the weight does not truely represent the level of drug/icd10 association because the size of cluster does affect to the level. High weight in big cluster falsely represents the high level of association because it also includes other icd10s in the cluster. The purify of the interested icd10 in the cluster is low even high weight which means the association level is low. This pattern is found when the drug is generally used. One drug is used in many icd10s (such as multivitamin, paracetamol, anti-histamine drugs). Therefore, the total number of icd10 in the cluster needs to be counted in the formula.

```
modified weight = number of icd10(x) in the cluster/(total number of icd10(x) x total number of icd10 in the cluster)
```

Then, the models were used to predict cluster from the test set and select top ten of icd10s (ranked by the modified weight). Group all predicted icd10s by txn and sum the weights of the same icd10 and rank the icd10s by aggregated weight again.


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
## Validation protocol

Because we have several teams to develop different approachs to predict ICD-10, we need to set up a standard protocol to validate across those approaches. We devide the validation protocol into to two parts: multilabel ranking metrics and TXN-based metrics.
1. Multilabel ranking metrics
[Multilabel ranking metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics) is a way to measure model performance when there is more than one correct answer per unit. The metrics include coverage error, label ranking average precision, and ranking loss.

1.1 Coverage error (CR) computes the average number of labels included in the final prediction that covers all true labels. The lower number means the better performance. The lowerest error means when the coverage error is equal to the average number of true label.

1.2 Label ranking average precision (AP) measure if the true labels are in the higer rank?. The value range is between 0 - 1 and 1 is the best.

1.3 Rank loss (RL) computes the avarage of the number of pairs that are incorrectly ordered. The value range is between 0 - 1 and 0 is the best.

2. TXN-based metrics
The model predict probability on each instance aggregated by TXN which cannot be directly measured by standard [evaluation metrics](https://www.analyticsvidhya.com/blog/2019/08/11-important-model-evaluation-error-metrics/). [Probability scoring methods](https://machinelearningmastery.com/how-to-score-probability-predictions-in-python/) are commonly used to measure the performance of probability prediction by calculating how close the probability of predicted class to actual class. These methods still treat the measurement per instance whereas our aim is to measure the performance as a group of instances aggregated by TXN. Therefore, we modified the standard evaluation metrics with probability scoring methods to define our evaluation approach.

***Example I:***
We aim to see how effective the model could predict correct diagnoses per visit. For example, the patient in this visit was diagnosed with 5 ICD-10s and prescribed with 8 drugs.

| Drug prescription | ICD-10 |
| :--- | :--- |
| ANapril Tab 5 mg | Angina pectoris, unspecified |
| Bestatin Tab   20 mg | Atherosclerotic heart disease |
| Dipazide Tab 5 mg *ยาเบาหวาน* | Essential (primary) hypertension |
| Furosemide Tab 40 mg | Hyperlipidaemia, unspecified |
| Hidil Cap 300 mg | Type 2 diabetes mellitus, without complications |
| Metoprolol 100 Stada Tab | |
| Monolin  60 SR CAP. | |
| Siamformet Tab 500 mg | |

Develop **ONE** model per **ONE** dataset (1 model for adm+reg, 1 model for dru+idru, 1 model for lab+ilab, and 1 model for rad+irad), independently. The predict the top twenty most likely ICD-10 per instance with probability weight (range from 0 - 1). For example, use the model to predict all instances in the same TXN (8 instances in this case) to predict 20 most likely ICD-10s. The predicted ICD-10s which match to the actual ICD-10 are presented in bold (except Angina pectoris, unspecified because it is not in the list).

| ICD-10 | Probability weight |
| :--- | :--- |
| Ischaemic cardiomyopathy  | 0.37 |
| Rheumatic mitral insufficiency  | 0.13 |
| Chronic viral hepatitis C  | 0.09 |
| Congestive heart failure  | 0.05 |
| Epilepsy, unspecified  | 0.05 |
| **Hyperlipidaemia, unspecified**  | 0.04 |
| Primary pulmonary hypertension  | 0.04 |
| Acute subendocardial myocardial infarction  | 0.03 |
| Diabetes mellitus, without complications  | 0.03 |
| Chronic renal failure, unspecified  | 0.03 |
| **Atherosclerotic heart disease**  | 0.02 |
| Tricuspid insufficiency  | 0.02 |
| Mitral stenosis  | 0.02 |
| Personal history of major surgery, not elsewhere classified  | 0.02 |
| **Type 2 diabetes mellitus, without complications**  | 0.01 |
| Systemic lupus erythematosus with involvement of organs and systems  | 0.01 |
| Atrial fibrillation and atrial flutter  | 0.01 |
| **Essential (primary) hypertension**  | 0.01 |
| Cholera  | 0.01 |
| Disorder of lipoprotein metabolism, unspecified  | 0.00 |

Count the number of true positive (TP), false positive (FP), true negative (TN), and false negative (FN), without considering the sequence. 

```
True positive (TP) = the number of actual ICD-10 that present in the list
```
```
False positive (FP) = the number of incorrect ICD-10 that present in the list before the last actual ICD-10 presenting in the list
```
```
True negative (TN) = the number of incorrect ICD-10 after the last actual ICD-10 presenting in the list
```
```
False negative (FN) = the number of actual ICD-10 that not present in the list
```


For top 20, TP = 4, FP = 14, TN = 2, and FN = 1.
Then, calculate accuracy, precision, recall, and F1 score (see the formulas [here](https://en.wikipedia.org/wiki/Confusion_matrix)).

Accuracy = (TP+TN)/N = (4+2)/20 = 0.40
Precision = TP/(TP+FP) = 4/(4+14) = 0.22
Recall = TP/(TP+FN) = 4/(4+1) = 0.80
F1 score = 2TP/(2TP+FP+FN) = (2x4)/((2X4)+14+1) = 0.35

Apply the same process to top 15, 10, and 5.

For top 15, TP = 3, FP = 12, TN = 0, and FN = 2.
For top 10, TP = 1, FP = 9, TN = 4, and FN = 4.
For top 5, TP = 0, FP = 5, TN = 0, and FN = 5.

Then, calculate accuracy, precision, recall, and F1 score again (process not shown).

| N | Accuracy | Precision | Recall | F1 score |
| :--- | :--- | :--- | :--- | :--- |
| 20 | 0.40 | 0.22 | 0.80 | 0.35 |
| 15 | 0.20 | 0.20 | 0.60 | 0.33 |
| 10 | 0.50 | 0.10 | 0.20 | 0.13 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 |

***Example II:***
The diagnosis if this case is Malignant neoplasm of cervix uteri, unspecified.

| ICD-10 | Probability weight |
| :--- | :--- |
| ***Malignant neoplasm of cervix uteri, unspecified*** | 0.23 |
| Traumatic subdural haemorrhage | 0.17 |
| Malignant neoplasm of nasopharynx, unspecified | 0.15 |
| Personal history of major surgery, not elsewhere classified | 0.06 |
| Malignant neoplasm of bronchus or lung, unspecified | 0.05 |
| Radiotherapy session | 0.05 |
| Malignant neoplasm of upper lobe, bronchus or lung | 0.05 |
| Malignant neoplasm of breast, unspecified | 0.05 |
| Secondary malignant neoplasm of lung | 0.03 |
| Malignant neoplasm of ovary | 0.03 |
| Malingnat neoplasm of exocervix | 0.02 |
| Secondary malignant neoplasm of liver | 0.02 |
| Chemotherapy session for neoplasm | 0.02 |
| Traumatic subdural haemorrhage: without open intracranial wound | 0.02 |
| Anaemia in neoplastic disease (C00-D48+) | 0.01 |
| Intrahepatic bile duct carcinoma | 0.01 |
| Agranulocytosis | 0.00 |
| Essential (primary) hypertension | 0.00 |
| Hypokalaemia | 0.00 |
| Acute posthaemorrhagic anaemia | 0.00 |

For top 20, TP = 1, FP = 0, TN = 19, and FN = 0.
For top 15, TP = 1, FP = 0, TN = 14, and FN = 0.
For top 10, TP = 1, FP = 0, TN = 9, and FN = 0.
For top 5, TP = 1, FP = 0, TN = 4, and FN = 0.

| N | Accuracy | Precision | Recall | F1 score |
| :--- | :--- | :--- | :--- | :--- |
| 20 | 1.00 | 1.00 | 1.00 | 1.00 |
| 15 | 1.00 | 1.00 | 1.00 | 1.00 |
| 10 | 1.00 | 1.00 | 1.00 | 1.00 |
| 5 | 1.00 | 1.00 | 1.00 | 1.00 |

***Example III:***
The diagnosis if this case is Parkinson's disease.

| ICD-10 | Probability weight |
| :--- | :--- |
| Dementia in Parkinson's disease (G20+) | 0.46 |
| ***Parkinson's disease*** | 0.29 |
| Sequelae of cerebral infarction | 0.20 |
| Other superficial injuries of lower leg | 0.01 |
| Mitral stenosis | 0.01 |
| Motorcycle rider injured in collision with car, pick-up truck or van, driver injured in traffic accident: during unspec activity | 0.00 |
| Motorcycle rider injured in noncollision transport accident, driver injured in traffic accident: during unspec activity | 0.00 |
| Acute lymphoblastic leukaemia | 0.00 |
| Cholera | 0.00 |
| Hyperlipidaemia, unspecified | 0.00 |
| Anaemia in other chronic diseases classified elsewhere | 0.00 |
| Urinary tract infection, site not specified | 0.00 |
| Agranulocytosis | 0.00 |
| Essential (primary) hypertension | 0.00 |
| Type 2 diabetes mellitus, without complications | 0.00 |
| Single live birth | 0.00 |
| Chemotherapy session for neoplasm | 0.00 |
| Atrial fibrillation and atrial flutter | 0.00 |
| Hyposmolality and hyponatraemia | 0.00 |
| Septicaemia, unspecified | 0.00 |

For top 20, TP = 1, FP = 1, TN = 18, and FN = 0.
For top 15, TP = 1, FP = 1, TN = 13, and FN = 0.
For top 10, TP = 1, FP = 1, TN = 8, and FN = 0.
For top 5, TP = 1, FP = 1, TN = 3, and FN = 0.

| N | Accuracy | Precision | Recall | F1 score |
| :--- | :--- | :--- | :--- | :--- |
| 20 | 0.95 | 0.50 | 1.00 | 0.67 |
| 15 | 0.93 | 0.50 | 1.00 | 0.67 |
| 10 | 0.90 | 0.50 | 1.00 | 0.67 |
| 5 | 0.80 | 0.50 | 1.00 | 0.67 |

After you finish validating all instances, then calculate weighted average accuracy, precision, recall, and F1 score.

```
weighted average accuracy = ((acc1 x n1) + (acc2 x n2) .... (accn x nn))/N
n = number of actual diagnoses in that txn, N = total number of actual diagnoses
```

weighted average accuracy top20 (A20) = ((0.40 x 5) + (1.00 x 1) + (0.95 x 1))/7 = 3.95/7 = 0.56
weighted average precision top20 (P20) = ((0.22 x 5) + (1.00 x 1) + (0.50 x 1))/7 = 2.60/7 = 0.37
weighted average recall top20 (R20) = ((0.80 x 5) + (1.00 x 1) + (0.95 x 1))/7 = 6.00/7 = 0.86
weighted average F1 score top20 (F20) = ((0.35 x 5) + (1.00 x 1) + (0.67 x 1))/7 = 3.32/7 = 0.47

## Result
10,000 instances (from the testset) were used to evaluate the multilabel ranking metrics.

| Dataset | Model | CR | AP | RL | A10 | P10 | R10 | F10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| reg & adm (vec) | MiniBatchKmean (n_cluster=15,000,top=10, RL) | 38,268 (total=38,969, avg_true_label=1.4)| 0.02 | 0.90 | 0.14 | 0.08 |
| dru & idru (vec) | MiniBatchKmean (n_cluster=15,000,top=10, RL) | 38,869 | 0.008 | 0.98 | 0.35 | 0.21 | 0.01 | 0.03 |
| dru (raw) | Alex secret (RL_3) | 42 | 0.55 | 0.016 |  |  |  | |
| dru (raw) | Alex secret (RL) | 180 | 0.46 | 0.012 |  |  |  | |

*RL = all icd-10 letters
*RL_3 = first 3 icd-10 letters

## Limitations
