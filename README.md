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
The amount of budget which can be claimed from the billing depends on the quality and completeness of the document. One major problem is the completeness of diagnosis (representing in ICD-10 code). The current process to complete the diagnosis in the document is a labour intersive work which requires physicians or technical coders to review medical records and manually enter a proper diagnosis code. Therefore, we see a potential benefit of machine learning application to automate this ICD-10 labelling process.
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

### Data preprocessing
All identification data such as name, surname, address, national identification, hospital number will be removed according to patient privacy. Data of interest include:
  * Demographic data such as date of birth, gender
  * History taking and physical examination (including discharge summary)
  * Laboratory and investigation reports
  * Medical prescription (investigation, drug)
  * ICD-10 (coded by a technical coder)
  
## Data analysis
Data from 2005 - 2016 are used to train machine learning models and data from 2017 are used to evaluate the models. We use overall accuracy, precision, recall, F-measure, and area under ROC curve to evaluate and compare predictive performance between models.

## Dataset
Data recorded between 2006 - 2019 from the electronic health records of Maharaj Nakhon Chiang Mai were deidentified and preprocessed. All data that could be potentially able to track back to an individual patient such as patients' name, surname, address, national identification number, address, phone number, hospital number were removed. We used **TXN** (a unique number representing a patient visit) to be a joining key. The dataset was divided into five groups.

1. Registration data
2. Admission data
3. Laboratory data
4. Radiological report data
5. Drug prescription data

### Registration data

The data cover the demographic information of patients visited at Maharaj Nakhon Chiang Mai hospital.
<details><summary>registration metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| age | numeric | age (year) |
| wt | numeric | weight (kg) |
| pulse | numeric | pulse rate (times/min) |
| resp | numeric | respiratory rate (times/min) |
| temp | numeric | body temperature (celcius) |
| sbp | numeric | systolic blood pressure (mmHg) |
| dbp | numeric | diastolic blood pressure (mmHg) |
| sex_f | binary | female |
| sex_m | binary | male |
| blood_a | binary | blood group A |
| blood_ab | binary | blood group AB |
| blood_b | binary | blood group B |
| blood_o | binary | blood group O |
| rh_n | binary | blood group Rh negative |
| rh_p | binary | blood group Rh positive |
| room_xxx | binary | Different type of rooms that patients visited: room_.,room_adm,room_adn,room_b,room_bs,room_che,room_chs,room_co,room_ctu,room_d,room_e,room_ecc,room_ent,room_er,room_eye,room_f,room_gmc,room_hp,room_hpro,room_ivt,room_j,room_jb,room_l,room_m,room_mdf,room_mdm,room_mdr,room_medsp,room_mnk,room_n,room_o,room_obbf,room_obthl,room_obui,room_opdslt,room_ors,room_p,room_ped,room_phyd,room_r,room_rado,room_re,room_rt,room_s,room_sgf,room_smi,room_spr,room_ss,room_t,room_ta,room_tb,room_tc,room_td,room_ttcm,room_vip,room_w,room_x,room_zwcmu,room_zxcmu,room_zycmu,room_zzcmu,room_ยา# |
| icd10 | text | ICD-10 code (diagnosis) |

</p>
</details>

### Admission data
### Laboratory data
### Radiological report data
### Drug prescription data

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
