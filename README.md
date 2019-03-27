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

The registration data is the demographic information of patients who visited (mostly outer patient department (OPD) cases) at Maharaj Nakhon Chiang Mai hospital.
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
| <room_xxx> | binary | Different room of outer patient departments that patients visited: room_.,room_adm,room_adn,room_b,room_bs,room_che,room_chs,room_co,room_ctu,room_d,room_e,room_ecc,room_ent,room_er,room_eye,room_f,room_gmc,room_hp,room_hpro,room_ivt,room_j,room_jb,room_l,room_m,room_mdf,room_mdm,room_mdr,room_medsp,room_mnk,room_n,room_o,room_obbf,room_obthl,room_obui,room_opdslt,room_ors,room_p,room_ped,room_phyd,room_r,room_rado,room_re,room_rt,room_s,room_sgf,room_smi,room_spr,room_ss,room_t,room_ta,room_tb,room_tc,room_td,room_ttcm,room_vip,room_w,room_x,room_zwcmu,room_zxcmu,room_zycmu,room_zzcmu,room_ยา# |
| icd10 | text | ICD-10 code (diagnosis) |

</p>
</details>

### Admission data

The admission data is the demographic information of patients who admitted to any internal wards (inner patient departments (IPD) cases) at Maharaj Nakhon Chiang Mai hospital.

<details><summary>admission metadata</summary>
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
| <room_xxx> | binary | Different inner patient departments that patients admitted: room_che,room_chem,room_cheo,room_crtf,room_crtm,room_eip,room_ens,room_ent,room_eye,room_eys,room_flo,room_gmc,room_gyn,room_gynn,room_kan,room_mdc,room_mdf,room_mdi,room_mdk,room_mdke,room_mdm,room_mdr,room_mdsc,room_mdt,room_mis,room_mnk,room_msi,room_mvdu,room_nim,room_nimi,room_obgi,room_obgr,room_obl,room_obs,room_obsv,room_ote,room_otf,room_oti,room_otm,room_otp,room_pd,room_pdi,room_pdn,room_pdni,room_pr,room_pri,room_psy,room_reh,room_roy,room_sgb,room_sge,room_sgf,room_sgic,room_sgig,room_sgin,room_sgm,room_sgn,room_sgp,room_sgt,room_ssf,room_ssm,room_wor,room_dc_che,room_dc_chem,room_dc_cheo,room_dc_crtf,room_dc_crtm,room_dc_eip,room_dc_ens,room_dc_ent,room_dc_eye,room_dc_eys,room_dc_flo,room_dc_gmc,room_dc_gyn,room_dc_gynn,room_dc_kan,room_dc_mdc,room_dc_mdf,room_dc_mdi,room_dc_mdm,room_dc_mdr,room_dc_mdsc,room_dc_mdt,room_dc_mis,room_dc_mnk,room_dc_msi,room_dc_mvdu,room_dc_nim,room_dc_nimi,room_dc_obgi,room_dc_obgr,room_dc_obl,room_dc_obs,room_dc_obsv,room_dc_ote,room_dc_otf,room_dc_oti,room_dc_otm,room_dc_otp,room_dc_pd,room_dc_pdi,room_dc_pdn,room_dc_pdni,room_dc_pr,room_dc_pri,room_dc_psy,room_dc_reh,room_dc_roy,room_dc_sgb,room_dc_sge,room_dc_sgf,room_dc_sgic,room_dc_sgig,room_dc_sgin,room_dc_sgm,room_dc_sgn,room_dc_sgp,room_dc_sgt,room_dc_ssf,room_dc_ssm,room_dc_wor |
| icd10 | text | ICD-10 code (diagnosis) |

</p>
</details>

### Laboratory data

#### L01

Laboratory findings of L01.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L01_0 | binary |  |
| L01_1 | binary |  |
| L01_2 | binary |  |
| L01_3 | binary |  |
| L01_4 | binary |  |
| L01_5 | binary |  |
| L01_6 | binary |  |
| L01_7 | binary |  |
| L01_8 | binary |  |
| L01_9 | binary |  |
| L01_10 | binary |  |
| L01_11 | binary |  |
| L01_12 | binary |  |
| L01_13 | binary |  |
| L01_14 | binary |  |
| L01_15 | binary |  |
| L01_16 | binary |  |
| L01_17 | binary |  |
| L01_18 | binary |  |
| L01_19 | binary |  |
| L01_20 | binary |  |
| L01_21 | binary |  |
| L01_22 | binary |  |
| L01_23 | binary |  |
| L01_24 | binary |  |
| L01_25 | binary |  |
| L01_26 | binary |  |
| L01_27 | binary |  |
| L01_28 | binary |  |
| L01_29 | binary |  |
| L01_30 | binary |  |
| L01_31 | binary |  |
| L01_32 | binary |  |
| L01_33 | binary |  |
| L01_34 | binary |  |
| L01_35 | binary |  |
| L01_36 | binary |  |
| L01_37 | binary |  |
| L01_38 | binary |  |
| L01_39 | binary |  |
| L01_40 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1901

Laboratory findings of L1901.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1901_0 | binary |  |
| L1901_1 | binary |  |
| L1901_2 | binary |  |
| L1901_3 | binary |  |
| L1901_4 | binary |  |
| L1901_5 | binary |  |
| L1901_6 | binary |  |
| L1901_7 | binary |  |
| L1901_8 | binary |  |
| L1901_9 | binary |  |
| L1901_10 | binary |  |
| L1901_11 | binary |  |
| L1901_12 | binary |  |
| L1901_13 | binary |  |
| L1901_14 | binary |  |
| L1901_15 | binary |  |
| L1901_16 | binary |  |
| L1901_17 | binary |  |
| L1901_18 | binary |  |
| L1901_19 | binary |  |
| L1901_20 | binary |  |
| L1901_21 | binary |  |
| L1901_22 | binary |  |
| L1901_23 | binary |  |
| L1901_24 | binary |  |
| L1901_25 | binary |  |
| L1901_26 | binary |  |
| L1901_27 | binary |  |
| L1901_28 | binary |  |
| L1901_29 | binary |  |
| L1901_30 | binary |  |
| L1901_31 | binary |  |
| L1901_32 | binary |  |
| L1901_33 | binary |  |
| L1901_34 | binary |  |
| L1901_35 | binary |  |
| L1901_36 | binary |  |
| L1901_37 | binary |  |
| L1901_38 | binary |  |
| L1901_39 | binary |  |
| L1901_40 | binary |  |
| L1901_41 | binary |  |
| L1901_42 | binary |  |
| L1901_43 | binary |  |
| L1901_44 | binary |  |
| L1901_45 | binary |  |
| L1901_46 | binary |  |
| L1901_47 | binary |  |
| L1901_48 | binary |  |
| L1901_49 | binary |  |
| L1901_50 | binary |  |
| L1901_51 | binary |  |
| L1901_52 | binary |  |
| L1901_53 | binary |  |
| L1901_54 | binary |  |
| L1901_55 | binary |  |
| L1901_56 | binary |  |
| L1901_57 | binary |  |
| L1901_58 | binary |  |
| L1901_59 | binary |  |
| L1901_60 | binary |  |
| L1901_61 | binary |  |
| L1901_62 | binary |  |
| L1901_63 | binary |  |
| L1901_64 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L0371

Laboratory findings of L0371.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L0371_0 | binary |  |
| L0371_1 | binary |  |
| L0371_2 | binary |  |
| L0371_3 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L0411

Laboratory findings of L0411.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L0411_0 | binary |  |
| L0411_1 | binary |  |
| L0411_2 | binary |  |
| L0411_3 | binary |  |
| L0411_4 | binary |  |
| L0411_5 | binary |  |
| L0411_6 | binary |  |
| L0411_7 | binary |  |
| L0411_8 | binary |  |
| L0411_9 | binary |  |
| L0411_10 | binary |  |
| L0411_11 | binary |  |
| L0411_12 | binary |  |
| L0411_13 | binary |  |
| L0411_14 | binary |  |
| L0411_15 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L10981

Laboratory findings of L10981.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L10981_0 | binary |  |
| L10981_1 | binary |  |
| L10981_2 | binary |  |
| L10981_3 | binary |  |
| L10981_4 | binary |  |
| L10981_5 | binary |  |
| L10981_6 | binary |  |
| L10981_7 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L531

Laboratory findings of L531.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L531_0 | binary |  |
| L531_1 | binary |  |
| L531_2 | binary |  |
| L531_3 | binary |  |
| L531_4 | binary |  |
| L531_5 | binary |  |
| L531_6 | binary |  |
| L531_7 | binary |  |
| L531_8 | binary |  |
| L531_9 | binary |  |
| L531_10 | binary |  |
| L531_11 | binary |  |
| L531_12 | binary |  |
| L531_13 | binary |  |
| L531_14 | binary |  |
| L531_15 | binary |  |
| L531_16 | binary |  |
| L531_17 | binary |  |
| L531_18 | binary |  |
| L531_19 | binary |  |
| L531_20 | binary |  |
| L531_21 | binary |  |
| L531_22 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1022

Laboratory findings of L1022.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1022_0 | binary |  |
| L1022_1 | binary |  |
| L1022_2 | binary |  |
| L1022_3 | binary |  |
| L1022_4 | binary |  |
| L1022_5 | binary |  |
| L1022_6 | binary |  |
| L1022_7 | binary |  |
| L1022_8 | binary |  |
| L1022_9 | binary |  |
| L1022_10 | binary |  |
| L1022_11 | binary |  |
| L1022_12 | binary |  |
| L1022_13 | binary |  |
| L1022_14 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L202

Laboratory findings of L202.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L202_0 | binary |  |
| L202_1 | binary |  |
| L202_2 | binary |  |
| L202_3 | binary |  |
| L202_4 | binary |  |
| L202_5 | binary |  |
| L202_6 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L10573

Laboratory findings of L10573.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L10573_0 | binary |  |
| L10573_1 | binary |  |
| L10573_2 | binary |  |
| L10573_3 | binary |  |
| L10573_4 | binary |  |
| L10573_5 | binary |  |
| L10573_6 | binary |  |
| L10573_7 | binary |  |
| L10573_8 | binary |  |
| L10573_9 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### B06

Laboratory findings of B06.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| B06_0 | binary |  |
| B06_1 | binary |  |
| B06_2 | binary |  |
| B06_3 | binary |  |
| B06_4 | binary |  |
| B06_5 | binary |  |
| B06_6 | binary |  |
| B06_7 | binary |  |
| B06_8 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L037

Laboratory findings of L037.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L037_0 | binary |  |
| L037_1 | binary |  |
| L037_2 | binary |  |
| L037_3 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L071

Laboratory findings of L071.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L071_0 | binary |  |
| L071_1 | binary |  |
| L071_2 | binary |  |
| L071_3 | binary |  |
| L071_4 | binary |  |
| L071_5 | binary |  |
| L071_6 | binary |  |
| L071_7 | binary |  |
| L071_8 | binary |  |
| L071_9 | binary |  |
| L071_10 | binary |  |
| L071_11 | binary |  |
| L071_12 | binary |  |
| L071_13 | binary |  |
| L071_14 | binary |  |
| L071_15 | binary |  |
| L071_16 | binary |  |
| L071_17 | binary |  |
| L071_18 | binary |  |
| L071_19 | binary |  |
| L071_20 | binary |  |
| L071_21 | binary |  |
| L071_22 | binary |  |
| L071_23 | binary |  |
| L071_24 | binary |  |
| L071_25 | binary |  |
| L071_26 | binary |  |
| L071_27 | binary |  |
| L071_28 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L10221

Laboratory findings of L10221.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L10221_0 | binary |  |
| L10221_1 | binary |  |
| L10221_2 | binary |  |
| L10221_3 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1914

Laboratory findings of L1914.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1914_0 | binary |  |
| L1914_1 | binary |  |
| L1914_2 | binary |  |
| L1914_3 | binary |  |
| L1914_4 | binary |  |
| L1914_5 | binary |  |
| L1914_6 | binary |  |
| L1914_7 | binary |  |
| L1914_8 | binary |  |
| L1914_9 | binary |  |
| L1914_10 | binary |  |
| L1914_11 | binary |  |
| L1914_12 | binary |  |
| L1914_13 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L58

Laboratory findings of L58.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L58_0 | binary |  |
| L58_1 | binary |  |
| L58_2 | binary |  |
| L58_3 | binary |  |
| L58_4 | binary |  |
| L58_5 | binary |  |
| L58_6 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L078

Laboratory findings of L078.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L078_0 | binary |  |
| L078_1 | binary |  |
| L078_2 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1001

Laboratory findings of L1001.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1001_0 | binary |  |
| L1001_1 | binary |  |
| L1001_2 | binary |  |
| L1001_3 | binary |  |
| L1001_4 | binary |  |
| L1001_5 | binary |  |
| L1001_6 | binary |  |
| L1001_7 | binary |  |
| L1001_8 | binary |  |
| L1001_9 | binary |  |
| L1001_10 | binary |  |
| L1001_11 | binary |  |
| L1001_12 | binary |  |
| L1001_13 | binary |  |
| L1001_14 | binary |  |
| L1001_15 | binary |  |
| L1001_16 | binary |  |
| L1001_17 | binary |  |
| L1001_18 | binary |  |
| L1001_19 | binary |  |
| L1001_20 | binary |  |
| L1001_21 | binary |  |
| L1001_22 | binary |  |
| L1001_23 | binary |  |
| L1001_24 | binary |  |
| L1001_25 | binary |  |
| L1001_26 | binary |  |
| L1001_27 | binary |  |
| L1001_28 | binary |  |
| L1001_29 | binary |  |
| L1001_30 | binary |  |
| L1001_31 | binary |  |
| L1001_32 | binary |  |
| L1001_33 | binary |  |
| L1001_34 | binary |  |
| L1001_35 | binary |  |
| L1001_36 | binary |  |
| L1001_37 | binary |  |
| L1001_38 | binary |  |
| L1001_39 | binary |  |
| L1001_40 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1081

Laboratory findings of L1081.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1081_0 | binary |  |
| L1081_1 | binary |  |
| L1081_2 | binary |  |
| L1081_3 | binary |  |
| L1081_4 | binary |  |
| L1081_5 | binary |  |
| L1081_6 | binary |  |
| L1081_7 | binary |  |
| L1081_8 | binary |  |
| L1081_9 | binary |  |
| L1081_10 | binary |  |
| L1081_11 | binary |  |
| L1081_12 | binary |  |
| L1081_13 | binary |  |
| L1081_14 | binary |  |
| L1081_15 | binary |  |
| L1081_16 | binary |  |
| L1081_17 | binary |  |
| L1081_18 | binary |  |
| L1081_19 | binary |  |
| L1081_20 | binary |  |
| L1081_21 | binary |  |
| L1081_22 | binary |  |
| L1081_23 | binary |  |
| L1081_24 | binary |  |
| L1081_25 | binary |  |
| L1081_26 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L022

Laboratory findings of L022.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L022_0 | binary |  |
| L022_1 | binary |  |
| L022_2 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L101763

Laboratory findings of L101763.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L101763_0 | binary |  |
| L101763_1 | binary |  |
| L101763_2 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L090

Laboratory findings of L090.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L090_0 | binary |  |
| L090_1 | binary |  |
| L090_2 | binary |  |
| L090_3 | binary |  |
| L090_4 | binary |  |
| L090_5 | binary |  |
| L090_6 | binary |  |
| L090_7 | binary |  |
| L090_8 | binary |  |
| L090_9 | binary |  |
| L090_10 | binary |  |
| L090_11 | binary |  |
| L090_12 | binary |  |
| L090_13 | binary |  |
| L090_14 | binary |  |
| L090_15 | binary |  |
| L090_16 | binary |  |
| L090_17 | binary |  |
| L090_18 | binary |  |
| L090_19 | binary |  |
| L090_20 | binary |  |
| L090_21 | binary |  |
| L090_22 | binary |  |
| L090_23 | binary |  |
| L090_24 | binary |  |
| L090_25 | binary |  |
| L090_26 | binary |  |
| L090_27 | binary |  |
| L090_28 | binary |  |
| L090_29 | binary |  |
| L090_30 | binary |  |
| L090_31 | binary |  |
| L090_32 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L029

Laboratory findings of L029.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L029_0 | binary |  |
| L029_1 | binary |  |
| L029_2 | binary |  |
| L029_3 | binary |  |
| L029_4 | binary |  |
| L029_5 | binary |  |
| L029_6 | binary |  |
| L029_7 | binary |  |
| L029_8 | binary |  |
| L029_9 | binary |  |
| L029_10 | binary |  |
| L029_11 | binary |  |
| L029_12 | binary |  |
| L029_13 | binary |  |
| L029_14 | binary |  |
| L029_15 | binary |  |
| L029_16 | binary |  |
| L029_17 | binary |  |
| L029_18 | binary |  |
| L029_19 | binary |  |
| L029_20 | binary |  |
| L029_21 | binary |  |
| L029_22 | binary |  |
| L029_23 | binary |  |
| L029_24 | binary |  |
| L029_25 | binary |  |
| L029_26 | binary |  |
| L029_27 | binary |  |
| L029_28 | binary |  |
| L029_29 | binary |  |
| L029_30 | binary |  |
| L029_31 | binary |  |
| L029_32 | binary |  |
| L029_33 | binary |  |
| L029_34 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L106011

Laboratory findings of L106011.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L106011_0 | binary |  |
| L106011_1 | binary |  |
| L106011_2 | binary |  |
| L106011_3 | binary |  |
| L106011_4 | binary |  |
| L106011_5 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L10961

Laboratory findings of L10961.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L10961_0 | binary |  |
| L10961_1 | binary |  |
| L10961_2 | binary |  |
| L10961_3 | binary |  |
| L10961_4 | binary |  |
| L10961_5 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L107018

Laboratory findings of L107018.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L107018_0 | binary |  |
| L107018_1 | binary |  |
| L107018_2 | binary |  |
| L107018_3 | binary |  |
| L107018_4 | binary |  |
| L107018_5 | binary |  |
| L107018_6 | binary |  |
| L107018_7 | binary |  |
| L107018_8 | binary |  |
| L107018_9 | binary |  |
| L107018_10 | binary |  |
| L107018_11 | binary |  |
| L107018_12 | binary |  |
| L107018_13 | binary |  |
| L107018_14 | binary |  |
| L107018_15 | binary |  |
| L107018_16 | binary |  |
| L107018_17 | binary |  |
| L107018_18 | binary |  |
| L107018_19 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1056221

Laboratory findings of L1056221.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1056221_0 | binary |  |
| L1056221_1 | binary |  |
| L1056221_2 | binary |  |
| L1056221_3 | binary |  |
| L1056221_4 | binary |  |
| L1056221_5 | binary |  |
| L1056221_6 | binary |  |
| L1056221_7 | binary |  |
| L1056221_8 | binary |  |
| L1056221_9 | binary |  |
| L1056221_10 | binary |  |
| L1056221_11 | binary |  |
| L1056221_12 | binary |  |
| L1056221_13 | binary |  |
| L1056221_14 | binary |  |
| L1056221_15 | binary |  |
| L1056221_16 | binary |  |
| L1056221_17 | binary |  |
| L1056221_18 | binary |  |
| L1056221_19 | binary |  |
| L1056221_20 | binary |  |
| L1056221_21 | binary |  |
| L1056221_22 | binary |  |
| L1056221_23 | binary |  |
| L1056221_24 | binary |  |
| L1056221_25 | binary |  |
| L1056221_26 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L093

Laboratory findings of L093.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L093_0 | binary |  |
| L093_1 | binary |  |
| L093_2 | binary |  |
| L093_3 | binary |  |
| L093_4 | binary |  |
| L093_5 | binary |  |
| L093_6 | binary |  |
| L093_7 | binary |  |
| L093_8 | binary |  |
| L093_9 | binary |  |
| L093_10 | binary |  |
| L093_11 | binary |  |
| L093_12 | binary |  |
| L093_13 | binary |  |
| L093_14 | binary |  |
| L093_15 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L10962

Laboratory findings of L10962.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L10962_0 | binary |  |
| L10962_1 | binary |  |
| L10962_2 | binary |  |
| L10962_3 | binary |  |
| L10962_4 | binary |  |
| L10962_5 | binary |  |
| L10962_6 | binary |  |
| L10962_7 | binary |  |
| L10962_8 | binary |  |
| L10962_9 | binary |  |
| L10962_10 | binary |  |
| L10962_11 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1902

Laboratory findings of L1902.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1902_0 | binary |  |
| L1902_1 | binary |  |
| L1902_2 | binary |  |
| L1902_3 | binary |  |
| L1902_4 | binary |  |
| L1902_5 | binary |  |
| L1902_6 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### GMCL001

Laboratory findings of GMCL001.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| GMCL001_0 | binary |  |
| GMCL001_1 | binary |  |
| GMCL001_2 | binary |  |
| GMCL001_3 | binary |  |
| GMCL001_4 | binary |  |
| GMCL001_5 | binary |  |
| GMCL001_6 | binary |  |
| GMCL001_7 | binary |  |
| GMCL001_8 | binary |  |
| GMCL001_9 | binary |  |
| GMCL001_10 | binary |  |
| GMCL001_11 | binary |  |
| GMCL001_12 | binary |  |
| GMCL001_13 | binary |  |
| GMCL001_14 | binary |  |
| GMCL001_15 | binary |  |
| GMCL001_16 | binary |  |
| GMCL001_17 | binary |  |
| GMCL001_18 | binary |  |
| GMCL001_19 | binary |  |
| GMCL001_20 | binary |  |
| GMCL001_21 | binary |  |
| GMCL001_22 | binary |  |
| GMCL001_23 | binary |  |
| GMCL001_24 | binary |  |
| GMCL001_25 | binary |  |
| GMCL001_26 | binary |  |
| GMCL001_27 | binary |  |
| GMCL001_28 | binary |  |
| GMCL001_29 | binary |  |
| GMCL001_30 | binary |  |
| GMCL001_31 | binary |  |
| GMCL001_32 | binary |  |
| GMCL001_33 | binary |  |
| GMCL001_34 | binary |  |
| GMCL001_35 | binary |  |
| GMCL001_36 | binary |  |
| GMCL001_37 | binary |  |
| GMCL001_38 | binary |  |
| GMCL001_39 | binary |  |
| GMCL001_40 | binary |  |
| GMCL001_41 | binary |  |
| GMCL001_42 | binary |  |
| GMCL001_43 | binary |  |
| GMCL001_44 | binary |  |
| GMCL001_45 | binary |  |
| GMCL001_46 | binary |  |
| GMCL001_47 | binary |  |
| GMCL001_48 | binary |  |
| GMCL001_49 | binary |  |
| GMCL001_50 | binary |  |
| GMCL001_51 | binary |  |
| GMCL001_52 | binary |  |
| GMCL001_53 | binary |  |
| GMCL001_54 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L10591

Laboratory findings of L10591.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L10591_0 | binary |  |
| L10591_1 | binary |  |
| L10591_2 | binary |  |
| L10591_3 | binary |  |
| L10591_4 | binary |  |
| L10591_5 | binary |  |
| L10591_6 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L61

Laboratory findings of L61.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L61_0 | binary |  |
| L61_1 | binary |  |
| L61_2 | binary |  |
| L61_3 | binary |  |
| L61_4 | binary |  |
| L61_5 | binary |  |
| L61_6 | binary |  |
| L61_7 | binary |  |
| L61_8 | binary |  |
| L61_9 | binary |  |
| L61_10 | binary |  |
| L61_11 | binary |  |
| L61_12 | binary |  |
| L61_13 | binary |  |
| L61_14 | binary |  |
| L61_15 | binary |  |
| L61_16 | binary |  |
| L61_17 | binary |  |
| L61_18 | binary |  |
| L61_19 | binary |  |
| L61_20 | binary |  |
| L61_21 | binary |  |
| L61_22 | binary |  |
| L61_23 | binary |  |
| L61_24 | binary |  |
| L61_25 | binary |  |
| L61_26 | binary |  |
| L61_27 | binary |  |
| L61_28 | binary |  |
| L61_29 | binary |  |
| L61_30 | binary |  |
| L61_31 | binary |  |
| L61_32 | binary |  |
| L61_33 | binary |  |
| L61_34 | binary |  |
| L61_35 | binary |  |
| L61_36 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L0221

Laboratory findings of L0221.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L0221_0 | binary |  |
| L0221_1 | binary |  |
| L0221_2 | binary |  |
| L0221_3 | binary |  |
| L0221_4 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L027

Laboratory findings of L027.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L027_0 | binary |  |
| L027_1 | binary |  |
| L027_2 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L10042

Laboratory findings of L10042.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L10042_0 | binary |  |
| L10042_1 | binary |  |
| L10042_2 | binary |  |
| L10042_3 | binary |  |
| L10042_4 | binary |  |
| L10042_5 | binary |  |
| L10042_6 | binary |  |
| L10042_7 | binary |  |
| L10042_8 | binary |  |
| L10042_9 | binary |  |
| L10042_10 | binary |  |
| L10042_11 | binary |  |
| L10042_12 | binary |  |
| L10042_13 | binary |  |
| L10042_14 | binary |  |
| L10042_15 | binary |  |
| L10042_16 | binary |  |
| L10042_17 | binary |  |
| L10042_18 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L422

Laboratory findings of L422.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L422_0 | binary |  |
| L422_1 | binary |  |
| L422_2 | binary |  |
| L422_3 | binary |  |
| L422_4 | binary |  |
| L422_5 | binary |  |
| L422_6 | binary |  |
| L422_7 | binary |  |
| L422_8 | binary |  |
| L422_9 | binary |  |
| L422_10 | binary |  |
| L422_11 | binary |  |
| L422_12 | binary |  |
| L422_13 | binary |  |
| L422_14 | binary |  |
| L422_15 | binary |  |
| L422_16 | binary |  |
| L422_17 | binary |  |
| L422_18 | binary |  |
| L422_19 | binary |  |
| L422_20 | binary |  |
| L422_21 | binary |  |
| L422_22 | binary |  |
| L422_23 | binary |  |
| L422_24 | binary |  |
| L422_25 | binary |  |
| L422_26 | binary |  |
| L422_27 | binary |  |
| L422_28 | binary |  |
| L422_29 | binary |  |
| L422_30 | binary |  |
| L422_31 | binary |  |
| L422_32 | binary |  |
| L422_33 | binary |  |
| L422_34 | binary |  |
| L422_35 | binary |  |
| L422_36 | binary |  |
| L422_37 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L074

Laboratory findings of L074.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L074_0 | binary |  |
| L074_1 | binary |  |
| L074_2 | binary |  |
| L074_3 | binary |  |
| L074_4 | binary |  |
| L074_5 | binary |  |
| L074_6 | binary |  |
| L074_7 | binary |  |
| L074_8 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L105933

Laboratory findings of L105933.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L105933_0 | binary |  |
| L105933_1 | binary |  |
| L105933_2 | binary |  |
| L105933_3 | binary |  |
| L105933_4 | binary |  |
| L105933_5 | binary |  |
| L105933_6 | binary |  |
| L105933_7 | binary |  |
| L105933_8 | binary |  |
| L105933_9 | binary |  |
| L105933_10 | binary |  |
| L105933_11 | binary |  |
| L105933_12 | binary |  |
| L105933_13 | binary |  |
| L105933_14 | binary |  |
| L105933_15 | binary |  |
| L105933_16 | binary |  |
| L105933_17 | binary |  |
| L105933_18 | binary |  |
| L105933_19 | binary |  |
| L105933_20 | binary |  |
| L105933_21 | binary |  |
| L105933_22 | binary |  |
| L105933_23 | binary |  |
| L105933_24 | binary |  |
| L105933_25 | binary |  |
| L105933_26 | binary |  |
| L105933_27 | binary |  |
| L105933_28 | binary |  |
| L105933_29 | binary |  |
| L105933_30 | binary |  |
| L105933_31 | binary |  |
| L105933_32 | binary |  |
| L105933_33 | binary |  |
| L105933_34 | binary |  |
| L105933_35 | binary |  |
| L105933_36 | binary |  |
| L105933_37 | binary |  |
| L105933_38 | binary |  |
| L105933_39 | binary |  |
| L105933_40 | binary |  |
| L105933_41 | binary |  |
| L105933_42 | binary |  |
| L105933_43 | binary |  |
| L105933_44 | binary |  |
| L105933_45 | binary |  |
| L105933_46 | binary |  |
| L105933_47 | binary |  |
| L105933_48 | binary |  |
| L105933_49 | binary |  |
| L105933_50 | binary |  |
| L105933_51 | binary |  |
| L105933_52 | binary |  |
| L105933_53 | binary |  |
| L105933_54 | binary |  |
| L105933_55 | binary |  |
| L105933_56 | binary |  |
| L105933_57 | binary |  |
| L105933_58 | binary |  |
| L105933_59 | binary |  |
| L105933_60 | binary |  |
| L105933_61 | binary |  |
| L105933_62 | binary |  |
| L105933_63 | binary |  |
| L105933_64 | binary |  |
| L105933_65 | binary |  |
| L105933_66 | binary |  |
| L105933_67 | binary |  |
| L105933_68 | binary |  |
| L105933_69 | binary |  |
| L105933_70 | binary |  |
| L105933_71 | binary |  |
| L105933_72 | binary |  |
| L105933_73 | binary |  |
| L105933_74 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1005

Laboratory findings of L1005.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1005_0 | binary |  |
| L1005_1 | binary |  |
| L1005_2 | binary |  |
| L1005_3 | binary |  |
| L1005_4 | binary |  |
| L1005_5 | binary |  |
| L1005_6 | binary |  |
| L1005_7 | binary |  |
| L1005_8 | binary |  |
| L1005_9 | binary |  |
| L1005_10 | binary |  |
| L1005_11 | binary |  |
| L1005_12 | binary |  |
| L1005_13 | binary |  |
| L1005_14 | binary |  |
| L1005_15 | binary |  |
| L1005_16 | binary |  |
| L1005_17 | binary |  |
| L1005_18 | binary |  |
| L1005_19 | binary |  |
| L1005_20 | binary |  |
| L1005_21 | binary |  |
| L1005_22 | binary |  |
| L1005_23 | binary |  |
| L1005_24 | binary |  |
| L1005_25 | binary |  |
| L1005_26 | binary |  |
| L1005_27 | binary |  |
| L1005_28 | binary |  |
| L1005_29 | binary |  |
| L1005_30 | binary |  |
| L1005_31 | binary |  |
| L1005_32 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L36

Laboratory findings of L36.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L36_0 | binary |  |
| L36_1 | binary |  |
| L36_2 | binary |  |
| L36_3 | binary |  |
| L36_4 | binary |  |
| L36_5 | binary |  |
| L36_6 | binary |  |
| L36_7 | binary |  |
| L36_8 | binary |  |
| L36_9 | binary |  |
| L36_10 | binary |  |
| L36_11 | binary |  |
| L36_12 | binary |  |
| L36_13 | binary |  |
| L36_14 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1903

Laboratory findings of L1903.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1903_0 | binary |  |
| L1903_1 | binary |  |
| L1903_2 | binary |  |
| L1903_3 | binary |  |
| L1903_4 | binary |  |
| L1903_5 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1904

Laboratory findings of L1904.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1904_0 | binary |  |
| L1904_1 | binary |  |
| L1904_2 | binary |  |
| L1904_3 | binary |  |
| L1904_4 | binary |  |
| L1904_5 | binary |  |
| L1904_6 | binary |  |
| L1904_7 | binary |  |
| L1904_8 | binary |  |
| L1904_9 | binary |  |
| L1904_10 | binary |  |
| L1904_11 | binary |  |
| L1904_12 | binary |  |
| L1904_13 | binary |  |
| L1904_14 | binary |  |
| L1904_15 | binary |  |
| L1904_16 | binary |  |
| L1904_17 | binary |  |
| L1904_18 | binary |  |
| L1904_19 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1030

Laboratory findings of L1030.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1030_0 | binary |  |
| L1030_1 | binary |  |
| L1030_2 | binary |  |
| L1030_3 | binary |  |
| L1030_4 | binary |  |
| L1030_5 | binary |  |
| L1030_6 | binary |  |
| L1030_7 | binary |  |
| L1030_8 | binary |  |
| L1030_9 | binary |  |
| L1030_10 | binary |  |
| L1030_11 | binary |  |
| L1030_12 | binary |  |
| L1030_13 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L5712

Laboratory findings of L5712.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L5712_0 | binary |  |
| L5712_1 | binary |  |
| L5712_2 | binary |  |
| L5712_3 | binary |  |
| L5712_4 | binary |  |
| L5712_5 | binary |  |
| L5712_6 | binary |  |
| L5712_7 | binary |  |
| L5712_8 | binary |  |
| L5712_9 | binary |  |
| L5712_10 | binary |  |
| L5712_11 | binary |  |
| L5712_12 | binary |  |
| L5712_13 | binary |  |
| L5712_14 | binary |  |
| L5712_15 | binary |  |
| L5712_16 | binary |  |
| L5712_17 | binary |  |
| L5712_18 | binary |  |
| L5712_19 | binary |  |
| L5712_20 | binary |  |
| L5712_21 | binary |  |
| L5712_22 | binary |  |
| L5712_23 | binary |  |
| L5712_24 | binary |  |
| L5712_25 | binary |  |
| L5712_26 | binary |  |
| L5712_27 | binary |  |
| L5712_28 | binary |  |
| L5712_29 | binary |  |
| L5712_30 | binary |  |
| L5712_31 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L107011

Laboratory findings of L107011.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L107011_0 | binary |  |
| L107011_1 | binary |  |
| L107011_2 | binary |  |
| L107011_3 | binary |  |
| L107011_4 | binary |  |
| L107011_5 | binary |  |
| L107011_6 | binary |  |
| L107011_7 | binary |  |
| L107011_8 | binary |  |
| L107011_9 | binary |  |
| L107011_10 | binary |  |
| L107011_11 | binary |  |
| L107011_12 | binary |  |
| L107011_13 | binary |  |
| L107011_14 | binary |  |
| L107011_15 | binary |  |
| L107011_16 | binary |  |
| L107011_17 | binary |  |
| L107011_18 | binary |  |
| L107011_19 | binary |  |
| L107011_20 | binary |  |
| L107011_21 | binary |  |
| L107011_22 | binary |  |
| L107011_23 | binary |  |
| L107011_24 | binary |  |
| L107011_25 | binary |  |
| L107011_26 | binary |  |
| L107011_27 | binary |  |
| L107011_28 | binary |  |
| L107011_29 | binary |  |
| L107011_30 | binary |  |
| L107011_31 | binary |  |
| L107011_32 | binary |  |
| L107011_33 | binary |  |
| L107011_34 | binary |  |
| L107011_35 | binary |  |
| L107011_36 | binary |  |
| L107011_37 | binary |  |
| L107011_38 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L421

Laboratory findings of L421.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L421_0 | binary |  |
| L421_1 | binary |  |
| L421_2 | binary |  |
| L421_3 | binary |  |
| L421_4 | binary |  |
| L421_5 | binary |  |
| L421_6 | binary |  |
| L421_7 | binary |  |
| L421_8 | binary |  |
| L421_9 | binary |  |
| L421_10 | binary |  |
| L421_11 | binary |  |
| L421_12 | binary |  |
| L421_13 | binary |  |
| L421_14 | binary |  |
| L421_15 | binary |  |
| L421_16 | binary |  |
| L421_17 | binary |  |
| L421_18 | binary |  |
| L421_19 | binary |  |
| L421_20 | binary |  |
| L421_21 | binary |  |
| L421_22 | binary |  |
| L421_23 | binary |  |
| L421_24 | binary |  |
| L421_25 | binary |  |
| L421_26 | binary |  |
| L421_27 | binary |  |
| L421_28 | binary |  |
| L421_29 | binary |  |
| L421_30 | binary |  |
| L421_31 | binary |  |
| L421_32 | binary |  |
| L421_33 | binary |  |
| L421_34 | binary |  |
| L421_35 | binary |  |
| L421_36 | binary |  |
| L421_37 | binary |  |
| L421_38 | binary |  |
| L421_39 | binary |  |
| L421_40 | binary |  |
| L421_41 | binary |  |
| L421_42 | binary |  |
| L421_43 | binary |  |
| L421_44 | binary |  |
| L421_45 | binary |  |
| L421_46 | binary |  |
| L421_47 | binary |  |
| L421_48 | binary |  |
| L421_49 | binary |  |
| L421_50 | binary |  |
| L421_51 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L54

Laboratory findings of L54.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L54_0 | binary |  |
| L54_1 | binary |  |
| L54_2 | binary |  |
| L54_3 | binary |  |
| L54_4 | binary |  |
| L54_5 | binary |  |
| L54_6 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1910

Laboratory findings of L1910.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1910_0 | binary |  |
| L1910_1 | binary |  |
| L1910_2 | binary |  |
| L1910_3 | binary |  |
| L1910_4 | binary |  |
| L1910_5 | binary |  |
| L1910_6 | binary |  |
| L1910_7 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### B13

Laboratory findings of B13.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| B13_0 | binary |  |
| B13_1 | binary |  |
| B13_2 | binary |  |
| B13_3 | binary |  |
| B13_4 | binary |  |
| B13_5 | binary |  |
| B13_6 | binary |  |
| B13_7 | binary |  |
| B13_8 | binary |  |
| B13_9 | binary |  |
| B13_10 | binary |  |
| B13_11 | binary |  |
| B13_12 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L551

Laboratory findings of L551.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L551_0 | binary |  |
| L551_1 | binary |  |
| L551_2 | binary |  |
| L551_3 | binary |  |
| L551_4 | binary |  |
| L551_5 | binary |  |
| L551_6 | binary |  |
| L551_7 | binary |  |
| L551_8 | binary |  |
| L551_9 | binary |  |
| L551_10 | binary |  |
| L551_11 | binary |  |
| L551_12 | binary |  |
| L551_13 | binary |  |
| L551_14 | binary |  |
| L551_15 | binary |  |
| L551_16 | binary |  |
| L551_17 | binary |  |
| L551_18 | binary |  |
| L551_19 | binary |  |
| L551_20 | binary |  |
| L551_21 | binary |  |
| L551_22 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L0421

Laboratory findings of L0421.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L0421_0 | binary |  |
| L0421_1 | binary |  |
| L0421_2 | binary |  |
| L0421_3 | binary |  |
| L0421_4 | binary |  |
| L0421_5 | binary |  |
| L0421_6 | binary |  |
| L0421_7 | binary |  |
| L0421_8 | binary |  |
| L0421_9 | binary |  |
| L0421_10 | binary |  |
| L0421_11 | binary |  |
| L0421_12 | binary |  |
| L0421_13 | binary |  |
| L0421_14 | binary |  |
| L0421_15 | binary |  |
| L0421_16 | binary |  |
| L0421_17 | binary |  |
| L0421_18 | binary |  |
| L0421_19 | binary |  |
| L0421_20 | binary |  |
| L0421_21 | binary |  |
| L0421_22 | binary |  |
| L0421_23 | binary |  |
| L0421_24 | binary |  |
| L0421_25 | binary |  |
| L0421_26 | binary |  |
| L0421_27 | binary |  |
| L0421_28 | binary |  |
| L0421_29 | binary |  |
| L0421_30 | binary |  |
| L0421_31 | binary |  |
| L0421_32 | binary |  |
| L0421_33 | binary |  |
| L0421_34 | binary |  |
| L0421_35 | binary |  |
| L0421_36 | binary |  |
| L0421_37 | binary |  |
| L0421_38 | binary |  |
| L0421_39 | binary |  |
| L0421_40 | binary |  |
| L0421_41 | binary |  |
| L0421_42 | binary |  |
| L0421_43 | binary |  |
| L0421_44 | binary |  |
| L0421_45 | binary |  |
| L0421_46 | binary |  |
| L0421_47 | binary |  |
| L0421_48 | binary |  |
| L0421_49 | binary |  |
| L0421_50 | binary |  |
| L0421_51 | binary |  |
| L0421_52 | binary |  |
| L0421_53 | binary |  |
| L0421_54 | binary |  |
| L0421_55 | binary |  |
| L0421_56 | binary |  |
| L0421_57 | binary |  |
| L0421_58 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L072

Laboratory findings of L072.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L072_0 | binary |  |
| L072_1 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1032

Laboratory findings of L1032.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1032_0 | binary |  |
| L1032_1 | binary |  |
| L1032_2 | binary |  |
| L1032_3 | binary |  |
| L1032_4 | binary |  |
| L1032_5 | binary |  |
| L1032_6 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L07

Laboratory findings of L07.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L07_0 | binary |  |
| L07_1 | binary |  |
| L07_2 | binary |  |
| L07_3 | binary |  |
| L07_4 | binary |  |
| L07_5 | binary |  |
| L07_6 | binary |  |
| L07_7 | binary |  |
| L07_8 | binary |  |
| L07_9 | binary |  |
| L07_10 | binary |  |
| L07_11 | binary |  |
| L07_12 | binary |  |
| L07_13 | binary |  |
| L07_14 | binary |  |
| L07_15 | binary |  |
| L07_16 | binary |  |
| L07_17 | binary |  |
| L07_18 | binary |  |
| L07_19 | binary |  |
| L07_20 | binary |  |
| L07_21 | binary |  |
| L07_22 | binary |  |
| L07_23 | binary |  |
| L07_24 | binary |  |
| L07_25 | binary |  |
| L07_26 | binary |  |
| L07_27 | binary |  |
| L07_28 | binary |  |
| L07_29 | binary |  |
| L07_30 | binary |  |
| L07_31 | binary |  |
| L07_32 | binary |  |
| L07_33 | binary |  |
| L07_34 | binary |  |
| L07_35 | binary |  |
| L07_36 | binary |  |
| L07_37 | binary |  |
| L07_38 | binary |  |
| L07_39 | binary |  |
| L07_40 | binary |  |
| L07_41 | binary |  |
| L07_42 | binary |  |
| L07_43 | binary |  |
| L07_44 | binary |  |
| L07_45 | binary |  |
| L07_46 | binary |  |
| L07_47 | binary |  |
| L07_48 | binary |  |
| L07_49 | binary |  |
| L07_50 | binary |  |
| L07_51 | binary |  |
| L07_52 | binary |  |
| L07_53 | binary |  |
| L07_54 | binary |  |
| L07_55 | binary |  |
| L07_56 | binary |  |
| L07_57 | binary |  |
| L07_58 | binary |  |
| L07_59 | binary |  |
| L07_60 | binary |  |
| L07_61 | binary |  |
| L07_62 | binary |  |
| L07_63 | binary |  |
| L07_64 | binary |  |
| L07_65 | binary |  |
| L07_66 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1911

Laboratory findings of L1911.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1911_0 | binary |  |
| L1911_1 | binary |  |
| L1911_2 | binary |  |
| L1911_3 | binary |  |
| L1911_4 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### B13.1

Laboratory findings of B13.1.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| B13.1_0 | binary |  |
| B13.1_1 | binary |  |
| B13.1_2 | binary |  |
| B13.1_3 | binary |  |
| B13.1_4 | binary |  |
| B13.1_5 | binary |  |
| B13.1_6 | binary |  |
| B13.1_7 | binary |  |
| B13.1_8 | binary |  |
| B13.1_9 | binary |  |
| B13.1_10 | binary |  |
| B13.1_11 | binary |  |
| B13.1_12 | binary |  |
| B13.1_13 | binary |  |
| B13.1_14 | binary |  |
| B13.1_15 | binary |  |
| B13.1_16 | binary |  |
| B13.1_17 | binary |  |
| B13.1_18 | binary |  |
| B13.1_19 | binary |  |
| B13.1_20 | binary |  |
| B13.1_21 | binary |  |
| B13.1_22 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L4301

Laboratory findings of L4301.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L4301_0 | binary |  |
| L4301_1 | binary |  |
| L4301_2 | binary |  |
| L4301_3 | binary |  |
| L4301_4 | binary |  |
| L4301_5 | binary |  |
| L4301_6 | binary |  |
| L4301_7 | binary |  |
| L4301_8 | binary |  |
| L4301_9 | binary |  |
| L4301_10 | binary |  |
| L4301_11 | binary |  |
| L4301_12 | binary |  |
| L4301_13 | binary |  |
| L4301_14 | binary |  |
| L4301_15 | binary |  |
| L4301_16 | binary |  |
| L4301_17 | binary |  |
| L4301_18 | binary |  |
| L4301_19 | binary |  |
| L4301_20 | binary |  |
| L4301_21 | binary |  |
| L4301_22 | binary |  |
| L4301_23 | binary |  |
| L4301_24 | binary |  |
| L4301_25 | binary |  |
| L4301_26 | binary |  |
| L4301_27 | binary |  |
| L4301_28 | binary |  |
| L4301_29 | binary |  |
| L4301_30 | binary |  |
| L4301_31 | binary |  |
| L4301_32 | binary |  |
| L4301_33 | binary |  |
| L4301_34 | binary |  |
| L4301_35 | binary |  |
| L4301_36 | binary |  |
| L4301_37 | binary |  |
| L4301_38 | binary |  |
| L4301_39 | binary |  |
| L4301_40 | binary |  |
| L4301_41 | binary |  |
| L4301_42 | binary |  |
| L4301_43 | binary |  |
| L4301_44 | binary |  |
| L4301_45 | binary |  |
| L4301_46 | binary |  |
| L4301_47 | binary |  |
| L4301_48 | binary |  |
| L4301_49 | binary |  |
| L4301_50 | binary |  |
| L4301_51 | binary |  |
| L4301_52 | binary |  |
| L4301_53 | binary |  |
| L4301_54 | binary |  |
| L4301_55 | binary |  |
| L4301_56 | binary |  |
| L4301_57 | binary |  |
| L4301_58 | binary |  |
| L4301_59 | binary |  |
| L4301_60 | binary |  |
| L4301_61 | binary |  |
| L4301_62 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L581

Laboratory findings of L581.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L581_0 | binary |  |
| L581_1 | binary |  |
| L581_2 | binary |  |
| L581_3 | binary |  |
| L581_4 | binary |  |
| L581_5 | binary |  |
| L581_6 | binary |  |
| L581_7 | binary |  |
| L581_8 | binary |  |
| L581_9 | binary |  |
| L581_10 | binary |  |
| L581_11 | binary |  |
| L581_12 | binary |  |
| L581_13 | binary |  |
| L581_14 | binary |  |
| L581_15 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L10502

Laboratory findings of L10502.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L10502_0 | binary |  |
| L10502_1 | binary |  |
| L10502_2 | binary |  |
| L10502_3 | binary |  |
| L10502_4 | binary |  |
| L10502_5 | binary |  |
| L10502_6 | binary |  |
| L10502_7 | binary |  |
| L10502_8 | binary |  |
| L10502_9 | binary |  |
| L10502_10 | binary |  |
| L10502_11 | binary |  |
| L10502_12 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L105621

Laboratory findings of L105621.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L105621_0 | binary |  |
| L105621_1 | binary |  |
| L105621_2 | binary |  |
| L105621_3 | binary |  |
| L105621_4 | binary |  |
| L105621_5 | binary |  |
| L105621_6 | binary |  |
| L105621_7 | binary |  |
| L105621_8 | binary |  |
| L105621_9 | binary |  |
| L105621_10 | binary |  |
| L105621_11 | binary |  |
| L105621_12 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L0261

Laboratory findings of L0261.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L0261_0 | binary |  |
| L0261_1 | binary |  |
| L0261_2 | binary |  |
| L0261_3 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L091

Laboratory findings of L091.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L091_0 | binary |  |
| L091_1 | binary |  |
| L091_2 | binary |  |
| L091_3 | binary |  |
| L091_4 | binary |  |
| L091_5 | binary |  |
| L091_6 | binary |  |
| L091_7 | binary |  |
| L091_8 | binary |  |
| L091_9 | binary |  |
| L091_10 | binary |  |
| L091_11 | binary |  |
| L091_12 | binary |  |
| L091_13 | binary |  |
| L091_14 | binary |  |
| L091_15 | binary |  |
| L091_16 | binary |  |
| L091_17 | binary |  |
| L091_18 | binary |  |
| L091_19 | binary |  |
| L091_20 | binary |  |
| L091_21 | binary |  |
| L091_22 | binary |  |
| L091_23 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L0414

Laboratory findings of L0414.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L0414_0 | binary |  |
| L0414_1 | binary |  |
| L0414_2 | binary |  |
| L0414_3 | binary |  |
| L0414_4 | binary |  |
| L0414_5 | binary |  |
| L0414_6 | binary |  |
| L0414_7 | binary |  |
| L0414_8 | binary |  |
| L0414_9 | binary |  |
| L0414_10 | binary |  |
| L0414_11 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L84

Laboratory findings of L84.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L84_0 | binary |  |
| L84_1 | binary |  |
| L84_2 | binary |  |
| L84_3 | binary |  |
| L84_4 | binary |  |
| L84_5 | binary |  |
| L84_6 | binary |  |
| L84_7 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1905

Laboratory findings of L1905.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1905_0 | binary |  |
| L1905_1 | binary |  |
| L1905_2 | binary |  |
| L1905_3 | binary |  |
| L1905_4 | binary |  |
| L1905_5 | binary |  |
| L1905_6 | binary |  |
| L1905_7 | binary |  |
| L1905_8 | binary |  |
| L1905_9 | binary |  |
| L1905_10 | binary |  |
| L1905_11 | binary |  |
| L1905_12 | binary |  |
| L1905_13 | binary |  |
| L1905_14 | binary |  |
| L1905_15 | binary |  |
| L1905_16 | binary |  |
| L1905_17 | binary |  |
| L1905_18 | binary |  |
| L1905_19 | binary |  |
| L1905_20 | binary |  |
| L1905_21 | binary |  |
| L1905_22 | binary |  |
| L1905_23 | binary |  |
| L1905_24 | binary |  |
| L1905_25 | binary |  |
| L1905_26 | binary |  |
| L1905_27 | binary |  |
| L1905_28 | binary |  |
| L1905_29 | binary |  |
| L1905_30 | binary |  |
| L1905_31 | binary |  |
| L1905_32 | binary |  |
| L1905_33 | binary |  |
| L1905_34 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L10041

Laboratory findings of L10041.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L10041_0 | binary |  |
| L10041_1 | binary |  |
| L10041_2 | binary |  |
| L10041_3 | binary |  |
| L10041_4 | binary |  |
| L10041_5 | binary |  |
| L10041_6 | binary |  |
| L10041_7 | binary |  |
| L10041_8 | binary |  |
| L10041_9 | binary |  |
| L10041_10 | binary |  |
| L10041_11 | binary |  |
| L10041_12 | binary |  |
| L10041_13 | binary |  |
| L10041_14 | binary |  |
| L10041_15 | binary |  |
| L10041_16 | binary |  |
| L10041_17 | binary |  |
| L10041_18 | binary |  |
| L10041_19 | binary |  |
| L10041_20 | binary |  |
| L10041_21 | binary |  |
| L10041_22 | binary |  |
| L10041_23 | binary |  |
| L10041_24 | binary |  |
| L10041_25 | binary |  |
| L10041_26 | binary |  |
| L10041_27 | binary |  |
| L10041_28 | binary |  |
| L10041_29 | binary |  |
| L10041_30 | binary |  |
| L10041_31 | binary |  |
| L10041_32 | binary |  |
| L10041_33 | binary |  |
| L10041_34 | binary |  |
| L10041_35 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1906

Laboratory findings of L1906.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1906_0 | binary |  |
| L1906_1 | binary |  |
| L1906_2 | binary |  |
| L1906_3 | binary |  |
| L1906_4 | binary |  |
| L1906_5 | binary |  |
| L1906_6 | binary |  |
| L1906_7 | binary |  |
| L1906_8 | binary |  |
| L1906_9 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L10044

Laboratory findings of L10044.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L10044_0 | binary |  |
| L10044_1 | binary |  |
| L10044_2 | binary |  |
| L10044_3 | binary |  |
| L10044_4 | binary |  |
| L10044_5 | binary |  |
| L10044_6 | binary |  |
| L10044_7 | binary |  |
| L10044_8 | binary |  |
| L10044_9 | binary |  |
| L10044_10 | binary |  |
| L10044_11 | binary |  |
| L10044_12 | binary |  |
| L10044_13 | binary |  |
| L10044_14 | binary |  |
| L10044_15 | binary |  |
| L10044_16 | binary |  |
| L10044_17 | binary |  |
| L10044_18 | binary |  |
| L10044_19 | binary |  |
| L10044_20 | binary |  |
| L10044_21 | binary |  |
| L10044_22 | binary |  |
| L10044_23 | binary |  |
| L10044_24 | binary |  |
| L10044_25 | binary |  |
| L10044_26 | binary |  |
| L10044_27 | binary |  |
| L10044_28 | binary |  |
| L10044_29 | binary |  |
| L10044_30 | binary |  |
| L10044_31 | binary |  |
| L10044_32 | binary |  |
| L10044_33 | binary |  |
| L10044_34 | binary |  |
| L10044_35 | binary |  |
| L10044_36 | binary |  |
| L10044_37 | binary |  |
| L10044_38 | binary |  |
| L10044_39 | binary |  |
| L10044_40 | binary |  |
| L10044_41 | binary |  |
| L10044_42 | binary |  |
| L10044_43 | binary |  |
| L10044_44 | binary |  |
| L10044_45 | binary |  |
| L10044_46 | binary |  |
| L10044_47 | binary |  |
| L10044_48 | binary |  |
| L10044_49 | binary |  |
| L10044_50 | binary |  |
| L10044_51 | binary |  |
| L10044_52 | binary |  |
| L10044_53 | binary |  |
| L10044_54 | binary |  |
| L10044_55 | binary |  |
| L10044_56 | binary |  |
| L10044_57 | binary |  |
| L10044_58 | binary |  |
| L10044_59 | binary |  |
| L10044_60 | binary |  |
| L10044_61 | binary |  |
| L10044_62 | binary |  |
| L10044_63 | binary |  |
| L10044_64 | binary |  |
| L10044_65 | binary |  |
| L10044_66 | binary |  |
| L10044_67 | binary |  |
| L10044_68 | binary |  |
| L10044_69 | binary |  |
| L10044_70 | binary |  |
| L10044_71 | binary |  |
| L10044_72 | binary |  |
| L10044_73 | binary |  |
| L10044_74 | binary |  |
| L10044_75 | binary |  |
| L10044_76 | binary |  |
| L10044_77 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L025

Laboratory findings of L025.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L025_0 | binary |  |
| L025_1 | binary |  |
| L025_2 | binary |  |
| L025_3 | binary |  |
| L025_4 | binary |  |
| L025_5 | binary |  |
| L025_6 | binary |  |
| L025_7 | binary |  |
| L025_8 | binary |  |
| L025_9 | binary |  |
| L025_10 | binary |  |
| L025_11 | binary |  |
| L025_12 | binary |  |
| L025_13 | binary |  |
| L025_14 | binary |  |
| L025_15 | binary |  |
| L025_16 | binary |  |
| L025_17 | binary |  |
| L025_18 | binary |  |
| L025_19 | binary |  |
| L025_20 | binary |  |
| L025_21 | binary |  |
| L025_22 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L2082

Laboratory findings of L2082.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L2082_0 | binary |  |
| L2082_1 | binary |  |
| L2082_2 | binary |  |
| L2082_3 | binary |  |
| L2082_4 | binary |  |
| L2082_5 | binary |  |
| L2082_6 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L083

Laboratory findings of L083.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L083_0 | binary |  |
| L083_1 | binary |  |
| L083_2 | binary |  |
| L083_3 | binary |  |
| L083_4 | binary |  |
| L083_5 | binary |  |
| L083_6 | binary |  |
| L083_7 | binary |  |
| L083_8 | binary |  |
| L083_9 | binary |  |
| L083_10 | binary |  |
| L083_11 | binary |  |
| L083_12 | binary |  |
| L083_13 | binary |  |
| L083_14 | binary |  |
| L083_15 | binary |  |
| L083_16 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L092

Laboratory findings of L092.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L092_0 | binary |  |
| L092_1 | binary |  |
| L092_2 | binary |  |
| L092_3 | binary |  |
| L092_4 | binary |  |
| L092_5 | binary |  |
| L092_6 | binary |  |
| L092_7 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L077

Laboratory findings of L077.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L077_0 | binary |  |
| L077_1 | binary |  |
| L077_2 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1084

Laboratory findings of L1084.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1084_0 | binary |  |
| L1084_1 | binary |  |
| L1084_2 | binary |  |
| L1084_3 | binary |  |
| L1084_4 | binary |  |
| L1084_5 | binary |  |
| L1084_6 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1052

Laboratory findings of L1052.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1052_0 | binary |  |
| L1052_1 | binary |  |
| L1052_2 | binary |  |
| L1052_3 | binary |  |
| L1052_4 | binary |  |
| L1052_5 | binary |  |
| L1052_6 | binary |  |
| L1052_7 | binary |  |
| L1052_8 | binary |  |
| L1052_9 | binary |  |
| L1052_10 | binary |  |
| L1052_11 | binary |  |
| L1052_12 | binary |  |
| L1052_13 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L09011

Laboratory findings of L09011.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L09011_0 | binary |  |
| L09011_1 | binary |  |
| L09011_2 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L073

Laboratory findings of L073.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L073_0 | binary |  |
| L073_1 | binary |  |
| L073_2 | binary |  |
| L073_3 | binary |  |
| L073_4 | binary |  |
| L073_5 | binary |  |
| L073_6 | binary |  |
| L073_7 | binary |  |
| L073_8 | binary |  |
| L073_9 | binary |  |
| L073_10 | binary |  |
| L073_11 | binary |  |
| L073_12 | binary |  |
| L073_13 | binary |  |
| L073_14 | binary |  |
| L073_15 | binary |  |
| L073_16 | binary |  |
| L073_17 | binary |  |
| L073_18 | binary |  |
| L073_19 | binary |  |
| L073_20 | binary |  |
| L073_21 | binary |  |
| L073_22 | binary |  |
| L073_23 | binary |  |
| L073_24 | binary |  |
| L073_25 | binary |  |
| L073_26 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L10501

Laboratory findings of L10501.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L10501_0 | binary |  |
| L10501_1 | binary |  |
| L10501_2 | binary |  |
| L10501_3 | binary |  |
| L10501_4 | binary |  |
| L10501_5 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L105932

Laboratory findings of L105932.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L105932_0 | binary |  |
| L105932_1 | binary |  |
| L105932_2 | binary |  |
| L105932_3 | binary |  |
| L105932_4 | binary |  |
| L105932_5 | binary |  |
| L105932_6 | binary |  |
| L105932_7 | binary |  |
| L105932_8 | binary |  |
| L105932_9 | binary |  |
| L105932_10 | binary |  |
| L105932_11 | binary |  |
| L105932_12 | binary |  |
| L105932_13 | binary |  |
| L105932_14 | binary |  |
| L105932_15 | binary |  |
| L105932_16 | binary |  |
| L105932_17 | binary |  |
| L105932_18 | binary |  |
| L105932_19 | binary |  |
| L105932_20 | binary |  |
| L105932_21 | binary |  |
| L105932_22 | binary |  |
| L105932_23 | binary |  |
| L105932_24 | binary |  |
| L105932_25 | binary |  |
| L105932_26 | binary |  |
| L105932_27 | binary |  |
| L105932_28 | binary |  |
| L105932_29 | binary |  |
| L105932_30 | binary |  |
| L105932_31 | binary |  |
| L105932_32 | binary |  |
| L105932_33 | binary |  |
| L105932_34 | binary |  |
| L105932_35 | binary |  |
| L105932_36 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1031

Laboratory findings of L1031.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1031_0 | binary |  |
| L1031_1 | binary |  |
| L1031_2 | binary |  |
| L1031_3 | binary |  |
| L1031_4 | binary |  |
| L1031_5 | binary |  |
| L1031_6 | binary |  |
| L1031_7 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L024

Laboratory findings of L024.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L024_0 | binary |  |
| L024_1 | binary |  |
| L024_2 | binary |  |
| L024_3 | binary |  |
| L024_4 | binary |  |
| L024_5 | binary |  |
| L024_6 | binary |  |
| L024_7 | binary |  |
| L024_8 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L02

Laboratory findings of L02.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L02_0 | binary |  |
| L02_1 | binary |  |
| L02_2 | binary |  |
| L02_3 | binary |  |
| L02_4 | binary |  |
| L02_5 | binary |  |
| L02_6 | binary |  |
| L02_7 | binary |  |
| L02_8 | binary |  |
| L02_9 | binary |  |
| L02_10 | binary |  |
| L02_11 | binary |  |
| L02_12 | binary |  |
| L02_13 | binary |  |
| L02_14 | binary |  |
| L02_15 | binary |  |
| L02_16 | binary |  |
| L02_17 | binary |  |
| L02_18 | binary |  |
| L02_19 | binary |  |
| L02_20 | binary |  |
| L02_21 | binary |  |
| L02_22 | binary |  |
| L02_23 | binary |  |
| L02_24 | binary |  |
| L02_25 | binary |  |
| L02_26 | binary |  |
| L02_27 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L10561

Laboratory findings of L10561.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L10561_0 | binary |  |
| L10561_1 | binary |  |
| L10561_2 | binary |  |
| L10561_3 | binary |  |
| L10561_4 | binary |  |
| L10561_5 | binary |  |
| L10561_6 | binary |  |
| L10561_7 | binary |  |
| L10561_8 | binary |  |
| L10561_9 | binary |  |
| L10561_10 | binary |  |
| L10561_11 | binary |  |
| L10561_12 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1040

Laboratory findings of L1040.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1040_0 | binary |  |
| L1040_1 | binary |  |
| L1040_2 | binary |  |
| L1040_3 | binary |  |
| L1040_4 | binary |  |
| L1040_5 | binary |  |
| L1040_6 | binary |  |
| L1040_7 | binary |  |
| L1040_8 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>

#### L1907

Laboratory findings of L1907.

<details><summary>admission metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| Unnamed: 0.1 | binary |  |
| L1907_0 | binary |  |
| L1907_1 | binary |  |
| L1907_2 | binary |  |
| L1907_3 | binary |  |
| L1907_4 | binary |  |
| L1907_5 | binary |  |
| L1907_6 | binary |  |
| L1907_7 | binary |  |
| L1907_8 | binary |  |
| L1907_9 | binary |  |
| L1907_10 | binary |  |
| icd10 | text | ICD-10 code (diagnosis) |
</p>
</details>


### Radiological report data
### Drug prescription data

The drug prescription data is the information of type of drugs which were prescribed to the patients.

<details><summary>drug prescription metadata</summary>
<p>

| Features | Types | Description |
| :--- | :--- | :--- |
| TXN | numeric | key identification for a patient visit |
| <drug_code> | text | ABAT01,ABAT02,ABAT03,ABAT04,ABAT05,ABAT06,ABAT07,ABIL01,ABIL02,ABIT01,ABIT02,ABIT03,ABIT04,ABIT05,ACAT01,ACC001,ACC002,ACCI01,ACCT01,ACCT02,ACE001,ACEC02,ACEH01,ACEH02,ACEI01,ACEI02,ACEI03,ACEL01,ACEL02,ACET01,ACII01,ACLI01,ACNEX1,ACT001,ACT002,ACT003,ACT004,ACTC01,ACTI01,ACTI02,ACTI03,ACTI04,ACTI05,ACTT02,ACTT03,ACTT04,ACTT05,ACTT06,ACTT07,ACTTX2,ACTTX3,ADAT01,ADAT02,ADAT03,ADAT04,ADAT05,ADDI01,ADDT01,ADEI01,ADRI01,ADRI02,ADRI03,ADRI04,ADRI05,ADRT01,ADRT02,ADVT01,AERE01,AERE02,AERL01,AERT01,AETI01,AETI02,AETI03,AFIT01,AGGT01,AGIL01,AGRT01,AGRT02,AIRL01,AIRT01,AKAI01,AKAI02,ALBI01,ALBI02,ALBI04,ALBI05,ALBI06,ALBL01,ALBL02,ALBLX1,ALBT01,ALBT02,ALCE02,ALCE03,ALCEX1,ALCEX2,ALCEX3,ALDE01,ALDT01,ALDT02,ALDT04,ALET01,ALET02,ALGT01,ALII01,ALII02,ALKI01,ALKI02,ALKT01,ALLH01,ALLI01,ALLI02,ALLT01,ALOE01,ALOE02,ALOI01,ALPE02,ALPE03,ALPI01,ALPI02,ALPI03,ALPT01,ALPT02,ALPT03,ALT001,ALT002,ALT003,ALT004,ALT005,ALT006,ALT007,ALUL04,ALUT01,ALUT02,ALUT03,ALWT01,AMAT01,AMAT02,AMAT03,AMAT04,AMBI01,AMBT01,AMIH01,AMIH02,AMIH03,AMII01,AMII02,AMII03,AMII04,AMII05,AMII07,AMII09,AMII12,AMII13,AMII14,AMIL02,AMIL04,AMIT01,AMIT02,AMIT03,AMKT01,AMLH01,AMLT01,AMLT02,AMME01,AMMEX1,AMOI01,AMOL01,AMOT03,AMOT05,AMOT06,AMOT07,AMOT08,AMPH01,AMPI01,AMPI02,AMPI03,AMPI04,AMPI05,AMPI06,AMPI07,AMPI08,AMPT01,AMTT01,ANAT01,ANAT02,ANAT03,ANAT05,ANAT06,ANDE01,ANDT01,ANDT02,ANDT03,ANEI01,ANEI02,ANEI03,ANG001,ANG002,ANG003,ANG004,ANGI01,ANGT01,ANK001,ANTI01,ANTI02,ANTI03,ANTI04,ANTI05,ANTI06,ANTI07,ANTI08,ANTI09,ANTI11,ANTI12,ANTI14,ANTI17,ANTI20,ANTT02,ANTT03,ANZI01,ANZI02,ANZI03,APOT01,APRT03,APRT04,APRT05,APUT01,AQU001,AQU002,AQUE01,ARAT01,ARAT02,ARAT03,ARCT01,ARCT02,ARCT03,ARCT04,AREI02,ARII01,ARIT01,ARIT02,ARIT03,ARIT04,ARIT05,ARM001,ARM002,ARM003,ARM004,AROT01,ARTE01,ARTE02,ARTEX1,ARTI01,ARTT02,ARTT03,ASAI01,ASAT01,ASCE01,ASCT01,ASK001,ASK002,ASK003,ASME01,ASML01,ASPT01,ASPT02,ASPT03,ASPT04,ASPT05,ASPT06,ASPT07,ASTE01,ASTI01,ASTT01,ASTT02,ATAL01,ATAT01,ATAT02,ATAT04,ATAT06,ATAT07,ATAT08,ATAT09,ATEH01,ATGI01,ATIT02,ATOE01,ATOT01,ATOT02,ATOT03,ATOT04,ATRE02,ATRI01,AUGI01,AUGI02,AUGL01,AUGL02,AUGL03,AUGT01,AUGT02,AUGT03,AUGT04,AVAE01,AVAI02,AVAT01,AVEI01,AVEI02,AVET01,AVF015,AVF316,AVOT01,AXOI01,AZAE01,AZAT01,AZIL01,AZIT01,AZIT02,AZOE01,AZYE01,AZYT01,BAC001,BAC002,BAC003,BAC004,BAC005,BAC006,BAC007,BACE01,BACE02,BACH01,BACI01,BACI02,BACT01,BAL001,BANE01,BANE02,BART01,BART02,BAST01,BCGI02,BCOT01,BECE01,BECT01,BEET01,BEFT01,BEFT02,BEFT03,BEFT04,BELL01,BELL02,BENE01,BENE03,BENE04,BENEX2,BENI03,BENI04,BENL01,BENL03,BENT01,BENT02,BENT03,BENT04,BEPE02,BERE01,BERE02,BERE03,BERI01,BERL01,BERL02,BERT01,BERT02,BERT03,BEST01,BEST02,BEST03,BETE01,BETE02,BETE03,BETE04,BETE05,BETE06,BETE08,BETEX1,BETEX2,BETEX3,BETEX4,BETI01,BETT01,BETT04,BETT05,BETT06,BETT07,BETT08,BEWT01,BEZT01,BEZT02,BFLI01,BIA001,BICI01,BILT01,BIO002,BIO003,BIO101,BIO103,BIO201,BIO203,BIOI01,BIOI02,BIOI03,BIOI04,BIOI05,BIOL01,BIOT01,BIOT02,BISI01,BISL01,BIST01,BIST02,BIST03,BLEI01,BLO001,BLOT01,BLOT02,BLOT03,BLOT04,BODL01,BONT01,BONT02,BONT04,BONT05,BOOL01,BOREX1,BOST01,BOTI01,BOTI02,BOTI03,BOTI31,BREE01,BRIE02,BRIE03,BRII01,BRIL02,BRIT01,BRIT02,BRIT03,BRIT04,BROL01,BROT01,BROT02,BSSE01,BSSE02,BUB001,BUB002,BUDE01,BUMI01,BUMI02,BUNE01,BUREX2,BUREX4,BUSI01,BUSI02,BUST01,BUST02,BYPT01,CACI01,CACT01,CADT01,CADT02,CADT03,CAEI01,CAFI01,CAFI02,CAFT01,CALE02,CALE03,CALEX2,CALH01,CALI03,CALI04,CALL01,CALT01,CALT04,CALT05,CALT06,CALT07,CALT08,CALT09,CALT10,CAMI01,CAMI02,CAN001,CANE02,CANE03,CANE04,CANI01,CANI02,CAP002,CAPE01,CAPE02,CAPT01,CAR001,CARH01,CARH02,CARI03,CARI04,CARL01,CARLX1,CART01,CART03,CART04,CART05,CART06,CART07,CART08,CART09,CART10,CART11,CART12,CART13,CASL01,CASLX1,CAST01,CAST02,CAST03,CAST04,CATE01,CAV001,CAV011,CAV012,CAV013,CAV014,CAV015,CAV016,CAVI01,CAVI02,CAVL01,CAVL02,CAVT01,CCLT01,CDOE01,CEFEX01,CEFH01,CEFH02,CEFI01,CEFI02,CEFI03,CEFI05,CEFI06,CEFI07,CEFI08,CEFI09,CEFI10,CEFI11,CEFL01,CEFT01,CEFT02,CELE01,CELT01,CELT02,CELT03,CELT04,CELT05,CELT06,CELT09,CENT01,CERI01,CERI02,CERI03,CERI04,CERI05,CERT01,CERT02,CERT03,CERT04,CEST01,CETT01,CHAT01,CHAT02,CHAT03,CHII01,CHLC01,CHLE01,CHLE02,CHLE03,CHLEX1,CHLEX2,CHLI01,CHLI02,CHLI03,CHLI04,CHLL01,CHLL02,CHLLX1,CHLT03,CHLT05,CHLT06,CIFI01,CIFT01,CILE01,CIPH01,CIPI01,CIPI02,CIPI03,CIPT01,CIPT02,CIPT03,CIPT04,CIPT05,CIPT06,CIPT07,CIRT01,CISI01,CISI02,CISI03,CISI04,CIST01,CLA004,CLA005,CLA006,CLA007,CLAI02,CLAI04,CLAL01,CLAL02,CLAT01,CLAT02,CLAT03,CLAT04,CLAT05,CLEI01,CLEI02,CLIE02,CLIEX1,CLIH01,CLII01,CLII02,CLIT02,CLOE01,CLOEX1,CLOI01,CLOI02,CLOI03,CLOI04,CLOI05,CLOL01,CLOT01,CLOT02,CLOT04,CLOT05,CLOT06,CLOT07,COAC02,COAEX3,COAEX4,COAEX5,COAEX6,COAT01,COAT02,COB001,COB002,CODT01,CODT02,CODT03,CODT04,CODT05,CODT06,COGI01,COL001,COL002,COL003,COL004,COL005,COL006,COL007,COL008,COL009,COL010,COL011,COL012,COL021,COL022,COL023,COL031,COL032,COL033,COL100,COLI01,COLI02,COLL01,COLT02,COLT03,COLT04,COLT05,COM001,COME01,COME02,COMT03,COMT04,COMT05,COMT06,COMT07,CONI01,CONT01,CONT03,CONT04,CONT05,CONT06,CONT08,COPT01,COPT02,COPT03,COPT04,CORC01,CORI01,CORI02,CORLX1,CORT01,CORT02,CORT03,CORT04,CORT05,CORT06,CORT07,COSE01,COSI01,COSI02,COT001,COT002,COTE01,COTL01,COTT01,COULX1,COVT01,COVT02,COVT04,COXT01,COZT01,COZT02,CRAE01,CRAE02,CRAI01,CRAI02,CRAT01,CRAT02,CRET01,CRET02,CRET03,CRIT01,CRU001,CRU002,CRU003,CRU004,CRU005,CRU006,CRU007,CRU008,CRU021,CRU022,CRU023,CUBI01,CUPT01,CURI01,CURL01,CURL02,CURT01,CUT002,CUT003,CVC007,CYCH01,CYCH02,CYCT01,CYCT02,CYCT03,CYMI01,CYMT01,CYMT02,CYRI01,CYRI02,CYTI01,CYTI02,CYTI03,CYTI04,CYTI05,CYTT01,D10S2,D10W03,D5SI01,D5SI02,D5SI03,D5SI04,D5SI05,D5SI06,D5SI07,D5WI01,D5WI02,D5WI03,D5WI04,D5WI05,D800,DACI01,DACI02,DACI03,DACI04,DACI05,DACI06,DACT01,DAFT02,DAFT03,DAFT04,DAFT05,DAIE01,DAIE03,DAIH01,DAKE01,DAKE03,DAKEX1,DAKT01,DALI01,DALI02,DALT01,DALT02,DANI02,DANT01,DAOT01,DART05,DAST01,DAST02,DAST04,DATI01,DAXI01,DAXI02,DAXT01,DBJ006,DBJ007,DBL001,DBL002,DBL101,DBL102,DEAT01,DECI01,DECT02,DEFT01,DEFT02,DEKL01,DELT01,DENT02,DEPI01,DEPI03,DEPI04,DEPL02,DEPT01,DEPT03,DERE01,DERE02,DERE03,DERE04,DESI01,DEST01,DETT01,DETT02,DETT03,DEWE01,DEXE01,DEXH01,DEXH02,DEXI01,DEXI02,DEXI03,DEXI04,DEXI05,DEXI06,DEXI07,DEXL01,DEXT01,DEXT02,DEXT03,DEXT04,DEXT05,DIAI01,DIAI02,DIAI04,DIAI05,DIAI06,DIAI07,DIAI08,DIAI09,DIAI12,DIAT01,DIAT02,DIAT03,DIAT04,DIAT06,DIAT07,DIAT08,DIAT09,DIAT10,DIAT11,DIAT13,DICL01,DICT01,DICT02,DICT03,DICT04,DIDT02,DIDT03,DIDT07,DIDT08,DIDT09,DIET01,DIFE01,DIFE02,DIFT03,DIGI01,DILI01,DILT01,DILT02,DILT03,DILT04,DIMI01,DIMI03,DIML01,DINT01,DIOT01,DIOT02,DIPC01,DIPE01,DIPE02,DIPE03,DIPE04,DIPE05,DIPI01,DIPI04,DIPI05,DIPI06,DIPI07,DIPI08,DIPI09,DIPI10,DIPI11,DIPL01,DIPT01,DIPT02,DIQE01,DIRT01,DISL01,DISL02,DIST02,DIST03,DIST04,DITI01,DITT01,DITT02,DIUT01,DIVE01,DIVI01,DIXI01,DIXT01,DOBI02,DOCI01,DOCI02,DOCI03,DOCI04,DOCI05,DOCI06,DOCT01,DOPE01,DOPI01,DOPI02,DOPL01,DOPT02,DORI01,DORI02,DORI03,DORT01,DORT02,DOSE01,DOSI01,DOST01,DOST02,DOUI01,DOUT01,DOXH01,DOXT01,DOZT01,DOZT02,DRAI01,DRAT01,DROI01,DROI02,DROI03,DROI04,DUIT01,DULE01,DULT01,DUO001,DUO002,DUO003,DUO004,DUOE01,DUOI01,DUOT01,DUOT02,DUPL01,DUPL02,DUPT01,DURE01,DURE02,DURE03,DURE04,DURE05,DURI01,DURI02,DURL01,DUST01,DUXT01,DYN003,DYN004,DYN005,DYNI01,DYSI01,DYSI02,EAS001,EBIL01,EBIT01,ECG002,EDAT01,EDAT02,EDII01,EDRL01,EDRT01,EDTEX1,EDUT01,EFAT01,EFAT02,EFAT03,EFAT04,EFAT05,EFAT08,EFAT09,EFAT10,EFEE01,EFET01,EFET02,EFFI01,EFFT01,EFFT02,EIFT01,ELA003,ELA004,ELA006,ELA013,ELA014,ELA016,ELA017,ELEL01,ELIE01,ELII01,ELIT01,ELME01,ELOE01,ELOE02,ELOE03,ELOI01,ELOI02,ELOT01,ELPLX3,EMEI01,EMET01,EMET02,EMLE01,EMTI01,EMTI02,EMTT01,EMUT01,ENAH01,ENAH02,ENAH03,ENAI01,ENAI02,ENBI01,ENCI01,ENCL01,ENCT01,END001,END002,ENDI01,ENDI02,ENDI03,ENDT01,ENFL01,ENTT01,ENTT02,EPHEX1,EPHEX2,EPHI01,EPOI02,EPRI01,EPRI02,EPRI03,EPRI05,EPRI06,EPRI08,ERAI01,ERBI01,ERGI01,ERYE01,ESBI01,ESIT01,ESMI01,ESMI02,ESPE01,ESPE02,ESPE03,ESPI01,ESPI02,ESPT01,ESST01,ESST02,ESTT01,ESTT02,ETAI01,ETHE01,ETHH01,ETHI01,ETHIX1,ETHT01,ETHT02,ETHT03,ETHT05,ETOE01,EU2E01,EUGT02,EUHT01,EURL01,EURL02,EURT01,EUTI01,EUTT01,EUTT02,EVOE01,EVOI01,EXEE01,EXEE02,EXEE03,EXET01,EXET02,EXET03,EXET05,EXET06,EXFT01,EXFT02,EXFT03,EXJT01,EXJT02,EXLT01,EXOE01,EXPI01,EXT003,EXT004,EXTI01,EXTI02,EYE114,EYE116,EYE118,EYLI01,EZET01,EZET02,FACI01,FACI02,FAMT01,FARI02,FARI03,FART01,FART05,FART06,FART07,FASI01,FASI02,FAVI01,FAVT03,FAVT04,FAZI01,FBCT01,FEBI01,FEBT01,FEE165,FEII01,FELT01,FELT02,FELT03,FELT04,FELT05,FELT06,FEMT01,FEMT02,FENE01,FENE02,FENE03,FENI01,FENI02,FENI03,FENI04,FENT01,FENT02,FENT03,FENT04,FENT05,FERL01,FERL02,FERL03,FERLX1,FERT01,FFUI01,FFUI02,FIBI01,FIBT01,FILI01,FIN001,FIRI02,FIRI04,FIRT01,FIVI01,FLAT02,FLAT03,FLAT04,FLEH01,FLEI01,FLEI02,FLEI03,FLEL01,FLET01,FLET02,FLIE01,FLIE02,FLOT01,FLTG01,FLUE01,FLUH01,FLUH02,FLUH03,FLUI01,FLUI02,FLUI03,FLUI04,FLUI05,FLUI06,FLUI08,FLUI09,FLUI11,FLUI12,FLUI20,FLUL01,FLUL02,FLUL03,FLUT01,FLUT04,FLUT05,FLUT06,FLUT08,FLUT10,FLUT11,FMLE01,FOBE01,FOL012,FOL014,FOL016,FOL018,FOL020,FOL021,FOL022,FOLH01,FOLT02,FOLT03,FOLT04,FOLT05,FORI01,FORI02,FORL01,FORT01,FORT02,FORT03,FOSE01,FOSI01,FOSI02,FOST02,FOST03,FOST04,FOST05,FOXT01,FRAI01,FRAI02,FRAI03,FRAI04,FREI01,FREI02,FUCE01,FUCE02,FUCE03,FUCE04,FUCT01,FUGT02,FULC01,FULT01,FUNI01,FUNT01,FUNT02,FURH01,FURI01,FURI02,FURI03,FURL01,FURT01,FURT02,FURT03,FYBL01,FYCT01,FYCT02,FYTI01,GABT01,GABT02,GABT03,GALT01,GALT02,GALT03,GAMI01,GAMI02,GANE01,GANE02,GANEX01,GANH01,GANT01,GARE03,GARI01,GASL01,GAST01,GAST02,GAST03,GAVL01,GAVT01,GELI01,GEMI01,GEMI02,GEMI03,GEMI04,GEMT01,GENE01,GENE02,GENE03,GENEX1,GENEX3,GENH01,GENI01,GENI02,GENI03,GENI04,GENI05,GERT01,GETI01,GILT01,GIOE01,GIOT01,GLAE03,GLAT02,GLIT02,GLIT03,GLIT04,GLIT05,GLIT06,GLUI02,GLUI03,GLUT01,GLUT02,GLUT03,GLUT04,GLUT07,GLUT08,GLYE01,GLYE02,GLYI01,GLYI02,GLYI03,GLYI04,GLYI05,GLYI06,GLYI07,GLYLX1,GLYT02,GLYT03,GOUT01,GPOT01,GPOT02,GPOT03,GPOT04,GPOT05,GPOT07,GPOT08,GPOT09,GPOT10,GPOT11,GPOT12,GPOT13,GPOT14,GPOT15,GPOT16,GRAI01,GRAI02,GRAT01,GRET01,GRIH01,GRIT03,GUAL01,GYNE01,GYNE02,HAE001,HAEI01,HAEI03,HALI01,HALI02,HALI03,HALI04,HALL01,HALT01,HALT03,HALT04,HALT05,HALT06,HALT08,HALT09,HALT10,HANEX1,HARI01,HARI02,HART01,HART02,HART03,HBII01,HBII02,HCOE01,HCQT01,HCTT01,HCTT08,HEAI01,HEAI02,HEAI03,HEAI04,HEBI01,HEM001,HEMI01,HEMI02,HEMI03,HEMI04,HEMI05,HEMI06,HEMI07,HEMI08,HEPI01,HEPI04,HEPI05,HEPI06,HEPI08,HEPL01,HEPL02,HEPL03,HEPT01,HEPT02,HERI01,HERI02,HERI03,HERT01,HERT02,HIAE01,HIAE02,HIBEX1,HIBEX2,HIDL01,HIDT01,HIG001,HIG002,HILT01,HIRE01,HIRE02,HISE01,HIZL01,HIZT01,HIZT02,HMII01,HOL001,HOLI01,HOLI02,HUMI01,HUMI04,HUMI07,HUMI09,HUMI10,HUMI11,HYAI01,HYCI01,HYCT01,HYCT02,HYDC02,HYDEX2,HYDEX3,HYDH01,HYDH02,HYDI01,HYDI02,HYDT01,HYDT02,HYDT03,HYDT04,HYDT05,HYDT06,HYOI01,HYP001,HYP002,HYP003,HYPE01,HYPT01,HYPT02,HYRI02,HYTT01,HYTT02,HYZT01,IAB001,IAB002,IAB003,IAB004,IAB005,IALI01,IBIL01,IBRL01,IBUT01,ICAT01,IDAT01,IGAI01,IKEE01,ILIE01,ILOI01,ILOL01,IMAT01,IMDT01,IMMI02,IMMI03,IMMI04,IMMI05,IMOI01,IMOT01,IMPE01,IMPE02,IMPT01,IMUT01,IMUT02,INDH01,INDI01,INDLX1,INDT01,INDT02,INDT04,INDT05,INDT07,INFE01,INFE03,INFE04,INFI01,INFI02,INFI03,INFI04,INFI05,INFT01,INHE01,INHT01,INJ001,INLT01,INNI01,INNI02,INNI03,INOI01,INSI01,INSI02,INSI03,INSI04,INSI05,INSI06,INSI07,INST01,INTI01,INTI02,INTI07,INTI08,INTI11,INTI12,INTI13,INTI14,INTI15,INTI16,INTI17,INTI18,INTT02,INTT03,INVI01,INVI02,INVI03,INVI04,INVT01,INVT02,INVT03,IOB001,IOB002,IOPI03,IPOE01,IRET01,IRET02,IRII01,IRII02,IRRT01,ISET01,ISMT01,ISOE01,ISOE02,ISOE03,ISOH01,ISOI01,ISOI02,ISOT01,ISOT02,ISOT03,ISOT04,ISOT05,ISOT07,ISOT08,ISUI01,ITRH01,ITRT01,ITRT02,IVC018,IVC020,IVC022,IVC024,IVGI01,IVGI02,IVII01,IVII02,IVII03,IVII04,IVS001,IVS002,IXEI01,JACE01,JADE01,JADT01,JAKT02,JAKT03,JANT01,JANT02,JANT03,JANT04,JART01,JART02,JART03,JEVI01,JEVI02,JEVI05,JEVI06,JUMT01,JUNL01,KABI01,KADI01,KALL01,KALT01,KALT02,KAME01,KAME02,KAME03,KAMI01,KANE01,KANI01,KANI02,KANI03,KANI04,KANI05,KANI06,KANI08,KAOL01,KAOL02,KAPT01,KAPT02,KARE01,KBA001,KBA002,KCLI01,KEFL01,KEFT02,KEFT03,KELT01,KEMI01,KEMI02,KEMI03,KENE01,KENE03,KENI02,KEPI01,KEPL01,KEPT01,KEPT02,KETE01,KETH01,KETH02,KETI01,KETI02,KETT01,KETT02,KEYI01,KIDI01,KIVT01,KIVT02,KIVT03,KIVT04,KLAI01,KLAL01,KLAT01,KLAT02,KLII01,KLII02,KLIT01,KLOT01,KNE001,KOMT01,KYJE02,KYTI01,KYTI02,KYTT01,LABI01,LACEX2,LACI01,LACI02,LAMT01,LAMT03,LAMT04,LAMT05,LAMT06,LAMT07,LAMT08,LAMT09,LAMT10,LAMT11,LAMT12,LAMT13,LAMT14,LAMT20,LANE01,LANI01,LANI02,LANI03,LANL01,LANT01,LANT02,LANT03,LASI01,LASI02,LASLX1,LAST01,LAST02,LAST03,LAST05,LAST06,LAST07,LAST08,LAST09,LCDEX1,LCDEX2,LCDEX3,LCDEX4,LEDT01,LEFI01,LEFI02,LEFT01,LEGT01,LEGT02,LENT01,LEST03,LETT01,LETT02,LEU001,LEU002,LEU003,LEU004,LEU100,LEUI01,LEUI02,LEUI04,LEUI08,LEUI09,LEUI10,LEUI13,LEUI14,LEUT02,LEVI01,LEVI02,LEVT01,LEVT02,LEXT01,LEXT02,LIBT02,LICT01,LIDI01,LIDI02,LIDI03,LINI01,LINI02,LINI03,LINT01,LIOI01,LIOT01,LIPI06,LIPI07,LIPI08,LIPI09,LIPI10,LIPI11,LIPT01,LIPT02,LIPT04,LIPT05,LIST01,LIST02,LITT01,LIVT01,LIVT02,LIXT01,LIXT02,LODI01,LODT01,LONT02,LOPL01,LOPT01,LOPT02,LOPT03,LOPT04,LOPT05,LOPT06,LOPT07,LORH01,LORT02,LORT03,LORT04,LORT05,LOSI01,LOST01,LOST02,LOTE01,LOW001,LOW002,LOW003,LOXT01,LSS001,LSS002,LSS003,LSS004,LSS005,LUBE01,LUCI01,LUE001,LUGLX1,LUME01,LUME02,LYRT01,LYRT02,LYRT03,LYRT04,MABI01,MABI02,MABI03,MABI04,MACI01,MADT01,MADT02,MADT03,MADT04,MAFT01,MAFT02,MAFT03,MAFT04,MAFT05,MAGI01,MAGI02,MAGI04,MAGI05,MAGLX2,MAGLX3,MAGT01,MAGT03,MAGT04,MAIT01,MALC01,MALT01,MAMT01,MAN001,MANI02,MANI03,MAPI01,MARE01,MARE02,MARI01,MARI03,MARI05,MARI07,MART01,MAWT01,MAXE01,MAXI01,MCTL01,MCTL02,MCTL03,MEBE01,MEBL01,MECE01,MECT01,MED003,MED004,MEDI01,MEDT01,MEDT02,MEDT03,MEFT01,MEGI01,MEGL01,MEGT01,MEGT03,MEIL01,MEIT01,MEIT02,MEIT03,MELE02,MELEX1,MELT01,MELT02,MELT03,MELT04,MEM001,MEM002,MENI01,MENL01,MEPI01,MEPL01,MEPT01,MEPT02,MEPT03,MERI01,MERI02,MERI03,MERT01,MERT02,MERT03,MERT04,MERT05,MERT06,MEST01,MEST03,METE01,METEX1,METEX10,METEX4,METEX5,METEX6,METH01,METH02,METH03,METI01,METI03,METI06,METI08,METI11,METI12,METL05,METLX5,METT01,METT02,METT03,METT04,METT05,METT06,METT07,METT08,METT09,METT10,METT11,METT12,METT14,METT15,MEVI01,MEVT02,MEVT03,MIAE01,MIAI01,MIC002,MICEX1,MICI01,MICI02,MICT01,MICT02,MICT03,MICT04,MIDI01,MIDI02,MILL01,MILL02,MIN001,MINE01,MINI01,MINT01,MINT02,MINT03,MINT06,MINT08,MIOI01,MIRI01,MIRI02,MIRT02,MIRT03,MIS001,MITH01,MITI01,MITI02,MITI03,MITI04,MIXI03,MIXI04,ML0586399,ML10586319,ML1586316,MMRI03,MOBE01,MOBI01,MOBT01,MODT02,MODT03,MONI01,MONI02,MONL01,MONT01,MONT02,MONT03,MONT04,MONT05,MONT06,MORI01,MORL01,MORLX2,MORLX3,MORT01,MOST01,MOTL01,MOTL02,MOTT01,MOTT02,MOZI01,MSESNG4AB,MSMONCSMP,MSOCPFTRB,MSOCPPTRB,MSOCSPTRB,MSOCSTRB,MSTT01,MSTT02,MSTT03,MUCL01,MUCL02,MUCL03,MUCT01,MULL02,MULL03,MULT01,MULT02,MUNL01,MV1586411,MV1586428,MV1602576,MXCALIC5FH,MXCAPL5FLD,MXCAPL5FMD,MXCAPL5FSD,MXCAPRFMD,MXCAPRLD,MXCAPRSD,MXCAQLSE,MXCARIC5FH,MXCATATP,MXCATATPT,MXCATLGBD,MXCATRVP,MXCATSAHVH,MXCATSMHVH,MXCATTFSPP,MXCATWSP,MXCAVA3XE,MXCBBTA,MXCBBTB,MXCBBTP,MXCBGATSL,MXCBGNPF5L,MXCBGPF2L,MXCBGPF5L,MXCBPL5FD,MXCCBBPRDN,MXCCBPMPRN,MXCCBSUVMQ,MXCCDOPGM,MXCCDOPGT,MXCCGPMVCM,MXCCIRFABM,MXCCMCTPC,MXCCMCTPM,MXCCOAVAE,MXCCOE,MXCCPTEWRM,MXCCPTEWVA,MXCCPTEWVM,MXCCPTPRCA,MXCCPTPRCM,MXCCTFGPD,MXCCTFGTR,MXCCVTD3M,MXCDRAPRM,MXCEBLZTCW,MXCECGTM,MXCECM1MQ,MXCECM2MQ,MXCECM3GT,MXCECM3MQ,MXCECM4MQ,MXCECMSMQ,MXCEFTSE,MXCEIMRARE,MXCETCTA,MXCETCTP,MXCETCTSE,MXCEVLRM1,MXCEVLRMT,MXCEWPPML,MXCEWPPS,MXCFMARTST,MXCFSAPTVK,MXCFSTBMT,MXCGWC032W,MXCGWC3514,MXCGWC3814,MXCGWJ3514,MXCGWJ3526,MXCGWJ3814,MXCGWS1826,MXCGWTRM25,MXCGWTRM35,MXCHCVAM,MXCHCVMM,MXCHDCM142,MXCHDCM181,MXCHDCM221,MXCHDDC162,MXCHDSG142,MXCHDSG201,MXCHDTC513,MXCHDTCF72,MXCHDTP551,MXCHDTP558,MXCHDTP7F1,MXCHDTP7F2,MXCHDVVVI,MXCHGHVTRC,MXCIHF4TF,MXCIHF9ET,MXCINREDL,MXCITATB34,MXCITATB40,MXCITDB6J,MXCITDB7J,MXCITDC411,MXCITDC510,MXCITDC511,MXCITDC51J,MXCITDC610,MXCITDC611,MXCITDC616,MXCITDC61J,MXCITDC623,MXCITDC711,MXCITDC712,MXCITDC722,MXCITDC723,MXCITDC810,MXCITDC811,MXCITDC81J,MXCITDC8E,MXCITDC8J,MXCITDCMP4,MXCITDCMP5,MXCITSCES,MXCJKBO4FD,MXCJKJIM5F,MXCJKL4F35,MXCJKL4F40,MXCJKL4H,MXCJKL5F35,MXCJKL5F40,MXCJKL5F50,MXCJKL5F5T,MXCJKL5J40,MXCJKL5T35,MXCJKL5T40,MXCJKL6F35,MXCJKL6F40,MXCJKR4F35,MXCJKR4F40,MXCJKR4F50,MXCJKR4H,MXCJKR4H40,MXCJKR4J35,MXCJKR5F35,MXCJKR5F40,MXCJKR5F50,MXCJKR5F5T,MXCJKR5J40,MXCJKR5T35,MXCJKR5T40,MXCJKR6F50,MXCJKSRC5F,MXCLMC6F10,MXCMAXILD,MXCMFSPAVM,MXCMLTST8H,MXCMNPVA,MXCMNPVM,MXCMOA,MXCMOP,MXCMRVA19E,MXCMRVM33E,MXCMTNVM,MXCMTPP510,MXCMTPP565,MXCMTPP580,MXCMTPP5A1,MXCNIH5F80,MXCNIH6F10,MXCNIH6F80,MXCOPCSM,MXCOXAHVK,MXCOXMHVK,MXCP3DARVN,MXCPDSCVOC,MXCPHVTNM,MXCPMEVEL,MXCPMVA19E,MXCPMVA21E,MXCPMVA23E,MXCPMVA27E,MXCPSCVOC,MXCPST72,MXCPSTB06,MXCPSTB12,MXCPSTB24,MXCPSTB48,MXCPSTB60,MXCPSTBMF1,MXCPSTBMF2,MXCPSTBMF3,MXCPSTBMF4,MXCPSTBMF6,MXCPSTBMF7,MXCPSTD600,MXCPSTDHPX,MXCPSTMF6,MXCPSTMM6,MXCPSTMM72,MXCPSTSD60,MXCPTC4F11,MXCPTC5F10,MXCPTC5F11,MXCPTC5FD,MXCPTCP5FW,MXCPTDTX,MXCPTSTD5F,MXCQPREMMQ,MXCQPRUCM,MXCRGAV,MXCSJRRTSJ,MXCSJVA19I,MXCSJVBCAT,MXCSJVBCMT,MXCSJVEPMT,MXCSJVM27I,MXCSLIKNI,MXCSRVA25K,MXCSRVM25K,MXCSWG4L7F,MXCTFTSTJ,MXCTGP,MXCTNCTPC,MXCTNCTPM,MXCTPRPL5S,MXCTPRPL6S,MXCTPRRPCW,MXCTRSIB,MXCUHPM,MXCVGGB45T,MXCVGGS30T,MXCXP4SCMQ,MXEARBTMX,MXEBPA4MI,MXEBPSTMI,MXECCKAC,MXECTRAQA,MXEHDKDAC,MXENCCIPAM,MXENCCIPIT,MXENTADM3,MXENTINT2,MXEOCBBMX,MXEPELFMX,MXEREUCJJ,MXESTLRBL,MXEXENGGI,MXEYAA000B,MXEYAA150B,MXEYAA160B,MXEYAA165B,MXEYAA170B,MXEYAA175B,MXEYAA180,MXEYAA185B,MXEYAA190B,MXEYAA195B,MXEYAA200B,MXEYAA205B,MXEYAA210B,MXEYAA215B,MXEYAA220B,MXEYAA225B,MXEYAA230B,MXEYAA235B,MXEYAA240B,MXEYAA245B,MXEYAA250B,MXEYAA255B,MXEYAA260B,MXEYAA265B,MXEYACRSE1,MXEYAMGCMV,MXEYAR4012,MXEYAR4014,MXEYAR4015,MXEYAR4016,MXEYAR4017,MXEYAR4018,MXEYAR4019,MXEYAR4020,MXEYAR4021,MXEYAR4022,MXEYAR4023,MXEYAR4024,MXEYAR4025,MXEYAR4026,MXEYAR4027,MXEYAR4028,MXEYAR4145,MXEYAR4155,MXEYAR4165,MXEYAR4175,MXEYAR4185,MXEYAR4195,MXEYAR4205,MXEYAR4215,MXEYAR4225,MXEYAR4235,MXEYAR4245,MXEYAR4255,MXEYAR4265,MXEYBVGIS,MXEYBVPPBL,MXEYCCKAGB,MXEYCF160M,MXEYCF165M,MXEYCF170M,MXEYCF175M,MXEYCF185M,MXEYCF190M,MXEYCF195M,MXEYCF200M,MXEYCF205M,MXEYCF210M,MXEYCF215M,MXEYCF220M,MXEYCF225M,MXEYCF230M,MXEYCF235M,MXEYCF240M,MXEYCF245M,MXEYCF260M,MXEYCM5512,MXEYCM5517,MXEYCM5518,MXEYCM5519,MXEYCM5520,MXEYCM5521,MXEYCM5522,MXEYCM5523,MXEYCM5524,MXEYCM5525,MXEYDKLZ,MXEYDL5175,MXEYDL5185,MXEYDL5195,MXEYDL5205,MXEYDL5215,MXEYDL5225,MXEYDL5235,MXEYDL5245,MXEYDL5255,MXEYDL5312,MXEYDL5317,MXEYDL5318,MXEYDL5319,MXEYDL5320,MXEYDL5321,MXEYDL5322,MXEYDL5323,MXEYDL5324,MXEYDL5325,MXEYDL6185,MXEYDL6195,MXEYDL6205,MXEYDL6215,MXEYDL6225,MXEYDL6235,MXEYDL6245,MXEYDL6255,MXEYDL6512,MXEYDL6513,MXEYDL6518,MXEYDL6519,MXEYDL6520,MXEYDL6521,MXEYDL6522,MXEYDL6523,MXEYDL6524,MXEYDL6525,MXEYDVTPBL,MXEYE5175,MXEYE5190,MXEYEOP160,MXEYEOP165,MXEYEOP175,MXEYEOP180,MXEYEOP185,MXEYEOP190,MXEYEOP195,MXEYEOP200,MXEYEOP205,MXEYEOP210,MXEYEOP215,MXEYEOP220,MXEYEOP225,MXEYEOP230,MXEYEOP240,MXEYEOP245,MXEYEOP250,MXEYEOP255,MXEYEOP260,MXEYEOPH55,MXEYEOS170,MXEYEOS175,MXEYEOS180,MXEYEOS185,MXEYEOS190,MXEYEOS195,MXEYEOS200,MXEYEOS205,MXEYEOS210,MXEYEOS215,MXEYEOS220,MXEYEOS225,MXEYEOS230,MXEYEOS235,MXEYEOS240,MXEYEOS245,MXEYEOSF65,MXEYEP205,MXEYEP230,MXEYEP235,MXEYEP240,MXEYEP5120,MXEYEP5160,MXEYEP5165,MXEYEP5170,MXEYEP5180,MXEYEP5185,MXEYEP5195,MXEYEP5200,MXEYEP5210,MXEYEP5215,MXEYEP5220,MXEYEP5225,MXEYEP5230,MXEYEP5235,MXEYEP5240,MXEYEP5245,MXEYEP5250,MXEYEP6000,MXEYEP6160,MXEYEP6165,MXEYEP6170,MXEYEP6175,MXEYEP6180,MXEYEP6185,MXEYEP6190,MXEYEP6195,MXEYEP6200,MXEYEP6205,MXEYEP6210,MXEYEP6215,MXEYEP6220,MXEYEP6225,MXEYEP6245,MXEYEP6250,MXEYEXPGFD,MXEYHAF200,MXEYHAF210,MXEYHAF220,MXEYITGLSS,MXEYLHSVBL,MXEYMA6006,MXEYMA60MA,MXEYMA6225,MXEYMA6245,MXEYMA6255,MXEYMZ3255,MXEYMZ6001,MXEYMZ6002,MXEYMZ6003,MXEYMZ605U,MXEYNRP0,MXEYNRP180,MXEYNRP185,MXEYNRP190,MXEYNRP195,MXEYNRP200,MXEYNRP205,MXEYNRP210,MXEYNRP215,MXEYNRP220,MXEYNRP225,MXEYNRP230,MXEYNRP235,MXEYNRP240,MXEYNRP250,MXEYOEP235,MXEYOFB220,MXEYOFY100,MXEYOFY110,MXEYOFY135,MXEYOFY165,MXEYOFY170,MXEYOFY175,MXEYOFY190,MXEYOFY195,MXEYOFY200,MXEYOFY205,MXEYOFY210,MXEYOFY215,MXEYOFY220,MXEYOFY225,MXEYOFY230,MXEYOFY235,MXEYOFY240,MXEYOFY250,MXEYOFYAFQ,MXEYOX1300,MXEYOX5700,MXEYOXHDB,MXEYPFORU,MXEYPVTCBL,MXEYR6150,MXEYR6155,MXEYR6160,MXEYR6165,MXEYR6170,MXEYR6175,MXEYR6180,MXEYR6185,MXEYR6190,MXEYR6195,MXEYR6200,MXEYR6205,MXEYR6210,MXEYR6215,MXEYR6220,MXEYR6225,MXEYR6230,MXEYR6235,MXEYR6240,MXEYR6245,MXEYR6250,MXEYR6255,MXEYR6260,MXEYRP155,MXEYRP160,MXEYRP170,MXEYRZMF2,MXEYSA6009,MXEYSA6010,MXEYSA6011,MXEYSA6012,MXEYSA6013,MXEYSA6014,MXEYSA6015,MXEYSA6016,MXEYSA6017,MXEYSA6018,MXEYSA6019,MXEYSA6020,MXEYSA6021,MXEYSA6022,MXEYSA6023,MXEYSA6024,MXEYSA6025,MXEYSA6026,MXEYSA6027,MXEYSA6028,MXEYSA6029,MXEYSA6034,MXEYSA6105,MXEYSA6115,MXEYSA6125,MXEYSA6135,MXEYSA6145,MXEYSA6155,MXEYSA6165,MXEYSA6175,MXEYSA6185,MXEYSA6195,MXEYSA6205,MXEYSA6215,MXEYSA6225,MXEYSA6235,MXEYSA6245,MXEYSA6255,MXEYSA6265,MXEYSA6275,MXEYSKAG30,MXEYSN6015,MXEYSN6016,MXEYSN6017,MXEYSN6018,MXEYSN6019,MXEYSN6020,MXEYSN6021,MXEYSN6022,MXEYSN6023,MXEYSN6024,MXEYSN6025,MXEYSN60WF,MXEYSN6155,MXEYSN6165,MXEYSN6175,MXEYSN6185,MXEYSN6195,MXEYSN6205,MXEYSN6215,MXEYSN6225,MXEYSN6235,MXEYSN6245,MXEYSN6255,MXEYSN6AD3,MXEYSN6T3,MXEYSN6T4,MXEYSN6T5,MXEYSNIQT3,MXEYSNIQT4,MXEYSNIQT5,MXEYSNIQT6,MXEYSNIQT7,MXEYSNIQT8,MXEYSNIQT9,MXEYTBCTRA,MXEYTFZ00,MXEYTFZ160,MXEYTFZ165,MXEYTFZ170,MXEYTFZ175,MXEYTFZ180,MXEYTFZ185,MXEYTFZ190,MXEYTFZ195,MXEYTFZ200,MXEYTFZ205,MXEYTFZ210,MXEYTFZ215,MXEYTFZ220,MXEYTFZ225,MXEYTFZ230,MXEYTFZ235,MXEYTFZ240,MXEYTFZ245,MXEYTFZ250,MXEYTFZ255,MXEYTFZ260,MXEYTFZ265,MXEYTFZ270,MXEYTMFZM9,MXEYTZC00,MXEYTZC165,MXEYTZC170,MXEYTZC175,MXEYTZC180,MXEYTZC185,MXEYTZC190,MXEYTZC195,MXEYTZC200,MXEYTZC205,MXEYTZC210,MXEYTZC215,MXEYTZC220,MXEYTZC225,MXEYTZC230,MXEYTZC235,MXEYTZC240,MXEYTZC245,MXEYTZC250,MXEYVTCBPP,MXEYVTIF23,MXEYVTTA23,MXEYVTTMAC,MXEYVTTMTP,MXEYVTTMTT,MXEYVTTP23,MXEYVTTPC2,MXEYVTTPC3,MXEYXLS180,MXEYXLS190,MXEYXLS195,MXEYXLS200,MXEYXLS205,MXEYXLS210,MXEYXLS215,MXEYXLS220,MXEYXLS225,MXEYXLS230,MXEYXLS240,MXEYXLS245,MXEYXLSTZO,MXGBATMFM,MXGBSCMTM,MXGBSUCTM,MXGUWESBT,MXGWCDSBT,MXIPPGBWT,MXIPPGCWT,MXLAPGGW,MXLAPLPTM,MXLAPTCWCT,MXLAPTCWFD,MXLAPTCWNN,MXLASBVSA,MXLASCVCD,MXLASTXPA,MXLASTXPP,MXLASTXVDE,MXLAVMDR,MXLBBCPDES,MXLBBLCID,MXLBBLMX,MXLBBLP,MXLBBLPI,MXLBBSTCBC,MXLBCTBLC,MXLBEPCFC,MXLBEPCS,MXLBEPTCN,MXLBEPTST,MXLBGDC,MXLBGDP,MXLBGDPTR,MXLBIDF,MXLBIVUSB,MXLBIVUSC,MXLBIVUSPS,MXLBMFBL,MXLBPTAW,MXLBPTCAW,MXLBRKNSJ,MXLBRTAV,MXLBRTB,MXLBRTBA2,MXLBRTBGW,MXLBSTP,MXLBSTPM,MXLBSTTXDE,MXLBSTXEDE,MXLBVVPBL,MXLCBAPJS,MXLCCVCVI,MXLCFMPMT,MXLCPLBAB,MXLCPSZV,MXLCRTLBT,MXLCRTPSJ,MXLCSZBVI,MXLCVSAVI,MXLDOICSJ,MXLECRTMT,MXLEDMPMT,MXLEPTABL,MXLGBLAB,MXLGBLES,MXLGBLID,MXLGBLP,MXLGBLPI,MXLGBLS,MXLGBLSAB,MXLGBLSID,MXLGCSTP,MXLGDPDEN6,MXLGICDL,MXLGICDPZ,MXLGICDS,MXLGICDVPD,MXLGICDVVR,MXLGIDFTPP,MXLGISHSJ,MXLGISUCS,MXLGPMIASR,MXLGPMIDR,MXLGPMIED,MXLGPMIEDR,MXLGPMIES,MXLGPMIESR,MXLGPMISE,MXLGPMISPS,MXLGPMIUDR,MXLGPMLSW,MXLGPTAW,MXLGPTCWC,MXLGRPCRT,MXLGRPCT2,MXLGRPMCT,MXLGRPML,MXLGSTPP,MXLGSTPX,MXLGSTPZ,MXLGSTTT,MXLGSTVS,MXLHBLAV,MXLHBLAVI,MXLHBLP,MXLHBLPI,MXLHPRPHE,MXLHPRPHP,MXLHSTP,MXLHSTRE,MXLIARC,MXLICMPC,MXLICPW,MXLIENSE,MXLIEPRD,MXLIEPREC,MXLIEPRO,MXLIEPRQ,MXLIPAICDV,MXLIPICDL,MXLIPIDAVR,MXLIPIDPMD,MXLIPIDPMV,MXLIPIEPHF,MXLIPMDBVF,MXLIPMDDD,MXLIPMDDDR,MXLIPMDIAD,MXLIPMDRID,MXLIPMDVAD,MXLIPMDVDR,MXLIPMLTD,MXLIPMLVQS,MXLIPMMSRA,MXLIPMVDDR,MXLIPMVILA,MXLIPMVIRA,MXLIPMVIRL,MXLIPMVVIC,MXLIPMVVIL,MXLIPMVVIR,MXLIPVTDR,MXLIRGTSJ,MXLISABC,MXLJAGPD,MXLJAJSFVG,MXLJBLAQ,MXLJBLCID,MXLJBLPAQ,MXLJBLPAQI,MXLJBLPI,MXLJEPCN,MXLJEPSF4,MXLJEPTGD,MXLJEPTN,MXLJEPTT,MXLJGDCVB,MXLJGDP,MXLJGDPJJ,MXLJOTIVCF,MXLJPBAM,MXLJPBAMI,MXLJPBAP,MXLJPBAPI,MXLJPBPFP,MXLJPBPFPI,MXLJPSPBAP,MXLJPSPGAM,MXLJPTAW,MXLJPTCAW,MXLJSEST,MXLJSETA2,MXLJSTCPS,MXLJSTPBS,MXLJSTSNPM,MXLKVESG,MXLLBLHPC,MXLLDESTP,MXLLPLBLC,MXLLPTBLC,MXLLSTPKCC,MXLMCRTMT,MXLMIBPTMC,MXLMINBIMS,MXLMSTCB,MXLMSTJDE,MXLMSTPC,MXLNAAASG,MXLNAILSG,MXLNBGWC,MXLNBLCID,MXLNBLPI,MXLNBLPL,MXLNBLPS,MXLNBLST,MXLNEAPC,MXLNGDC,MXLNGDP,MXLNGDPMT,MXLNICDGL,MXLNICDL,MXLNICDMQ,MXLNPITSJ,MXLNPMEPDR,MXLNPMKP7,MXLNPMKP9,MXLNPML,MXLNPMLE,MXLNPMMBT,MXLNPMS10,MXLNPMS20,MXLNPMSD20,MXLNPMSDR,MXLNPMSDR2,MXLNPMSEDR,MXLNPMSES1,MXLNPMSESR,MXLNPMSSD,MXLNPMSSR,MXLNPMSVD3,MXLNPTCAW,MXLNRPMIS,MXLNRPML,MXLNST,MXLNSTED,MXLNSTENM,MXLOSTMTM,MXLOSTTGM,MXLPBMFDES,MXLPCTOPM,MXLPCTOPP,MXLPISSJ,MXLPMSRBT,MXLPSTG,MXLPSTP,MXLPTCAAS,MXLSAVCD,MXLTBLCID,MXLTBLHYT,MXLTBLP,MXLTBLPI,MXLTBLPJJ,MXLTBLTRM,MXLTGDGCS,MXLTHPR67,MXLTHPRTR,MXLTHTGC5V,MXLTHTGC67,MXLTMADSAB,MXLTMASO,MXLTOSTFB2,MXLTOSTFBD,MXLTPMC,MXLTPMCA2,MXLTPTAW,MXLTPTCACW,MXLTPTCPR,MXLTSTNBDE,MXLTSTP,MXLTSTTRM,MXLTTBAPC,MXLTTSAB1,MXLVALVUL,MXLVGTVMF,MXLWAPMBB,MXLWAPMBS,MXLWCAASD,MXLWCAPDA,MXLWCASD1,MXLWCASDO,MXLWCASVC,MXLWCAVI,MXLWCPDAO,MXLWCSB,MXMABRMIS,MXMACAG10B,MXMACAG15B,MXMACAG20B,MXMACCMTEL,MXMACEKH,MXMAGP1010,MXMAGSM023,MXMAGSM101,MXMAKNFB,MXMAKNGB,MXMALVA75S,MXMALVAS12,MXMALVSD10,MXMAPAPCRS,MXMAPAPFPA,MXMAPAPGMC,MXMAPAPREM,MXMAPAPS9E,MXMAPAPSWM,MXMAPAPVTA,MXMAPCKBT,MXMAQAG10D,MXMAQAG20D,MXMAQPHMH,MXMATFC2D,MXMATRBC,MXMBOSPD6D,MXMBTRFGBP,MXMBTRFTPM,MXMCC1215S,MXMCC312S,MXMCC612S,MXMCCFD,MXMCDBC13M,MXMCDBC33M,MXMCNSBF3M,MXMCODSJIJ,MXMCPAMCM,MXMCPAMCP,MXMCPAMPC,MXMCPAMSM,MXMCPAP2EW,MXMCPAPCRS,MXMCPAPDEV,MXMCPAPFPN,MXMCPAPGMC,MXMCPAPPTA,MXMCPAPREM,MXMCPAPRPS,MXMCPAPSIC,MXMCPAPTGS,MXMCPMFSF,MXMCPMTA,MXMCSSAE,MXMCSSPE,MXMCSSS2BH,MXMCSSS4BH,MXMCSTCE,MXMEBCTA2,MXMEBCTNC,MXMERCLS20,MXMFITSLPB,MXMFMCSRM,MXMFSHM05B,MXMFSHSN05,MXMGN22PH,MXMHBTSMA,MXMHNRSTCP,MXMHPFD,MXMHSSTCP,MXMIMRTHDM,MXMIMRTHSD,MXMISHD,MXMKESSTI,MXMKFSTI,MXMKFTTI,MXMLKCRFB,MXMMTLC375,MXMNARCA2,MXMNSPF8EM,MXMNSPS8EM,MXMNVLSMJ,MXMONCSMA,MXMOXSS25,MXMPCCSBL,MXMPMSCDVP,MXMRSTDMM,MXMSBPTEQ,MXMSESTCVD,MXMSPFBJ,MXMTEDLLB,MXMTEDLRB,MXMTEDLSB,MXMTEDMLB,MXMTEDMRB,MXMTEDMSB,MXMTEDSLB,MXMTEDSRB,MXMTEDSSB,MXMTFHTI,MXMTPES780,MXMTTTF,MXMUGCAUG,MXMUGSSD12,MXMUGSSD20,MXMUGT1010,MXMUNMTD,MXMUROFD,MXMVDHTI,MXMVLMTTST,MXMVPSU,MXMVRIJT23,MXMWCASM50,MXMWCASMUC,MXMWCCPUUC,MXMWCCVRSN,MXMWCFS8IS,MXMWCFS8UC,MXMWCFS9IS,MXMWCFS9UC,MXMWCOT,MXMWCSBGSN,MXMWCSBGUC,MXMWCSYAUC,MXMWCSYKSN,MXMWCSYKUC,MXMWCSZCSP,MXMWCSZCUC,MXMWCT,MXNAHRX18M,MXNAHRX25M,MXNAQMH03E,MXNAQMH07E,MXNAQMH12E,MXNCCF20M,MXNCCF40M,MXNDBLMCS3,MXNDBLMCSL,MXNES210SV,MXNHMCNTR,MXNMARDZAS,MXNMARECAS,MXNMARHACC,MXNMARHIEC,MXNMHKC135,MXNMRDCMT,MXNPFMCPBT,MXNPHDFSFN,MXNPMFLOM,MXNPMFP1SF,MXOAHPNV,MXOAQCFCV,MXOAQCNCV,MXOAQCSCV,MXOAQCSHC,MXOAQCSLC,MXOAQNACV,MXOASPNV,MXOASRNV,MXOATMH41B,MXOATMH48B,MXOATMH49B,MXOATMH51B,MXOATMH51D,MXOBF1CCNV,MXOBRBLNV,MXOBRBSNV,MXOCBBYT,MXOCFFBNV,MXOCWSPNV,MXOEBRGNV,MXOEDASSSP,MXOFTRADMN,MXOFTRAKXL,MXOFTREBXL,MXOFTREWS,MXOFTRHNXL,MXOFTRSPWL,MXOFTRSPWR,MXOFTRTSBJ,MXOFTRWSD,MXOHCHT,MXOHIPDSSP,MXOISETDHT,MXOJWBEXYT,MXOJWBMA,MXOKOSPSS,MXOKRGLNV,MXOKRGNVL,MXOKSPBJC,MXOKSPFXLD,MXOKSPSEF,MXOKSPSS,MXOMCRMC,MXOMLSLMM,MXOMPCLM,MXOMPCMM,MXOMPCSM,MXONCTWNV,MXOOCMTG1,MXOP3123M,MXOP4303M,MXOP5453M,MXOP6303M,MXOPFPKSNV,MXOPMTWNV,MXOQCT,MXOQCTISM,MXOQUICML,MXOSC343M,MXOSC443M,MXOSFAKSPS,MXOSFBSPSS,MXOSFHKSLS,MXOSFHKSPS,MXOSFK,MXOSFLSSPS,MXOSFTSPS,MXOSFWSPSS,MXOSGSCML,MXOSHKL,MXOSOMIBYT,MXOTCHCNP,MXOTGTSNP,MXOTLBYT,MXOTNEFD,MXOUTSLNV,MXRAFOCDM,MXRASBLPC,MXRFPSLPC,MXRFS609U,MXRKY87PM,MXRKY87PU,MXRKY903PM,MXRKY903PU,MXRKY98PM,MXRKY98PU,MXRRPMSLP,MXRSYIVSN,MXRSYIVSU,MXSADMMTM,MXSADMMTN,MXSADMVLP,MXSAKNTLB,MXSAKNTSB,MXSALAIS,MXSALNIVC,MXSALNIVP,MXSAMBPFK,MXSANRSC11,MXSANRSC18,MXSANRSCL1,MXSANRSCL2,MXSANRSCL8,MXSANRSCS1,MXSANRSCS2,MXSAOWRWG,MXSAOWRXLW,MXSAOWRXSW,MXSASAIS,MXSASNCTMT,MXSATFACAT,MXSATRCCAT,MXSATSSCAT,MXSATSSCFS,MXSAXLAIS,MXSBLMSTED,MXSBLSETTB,MXSBTPBMP,MXSBTPNMP,MXSBTPSMP,MXSCDHUCST,MXSCEEAUCS,MXSCEEAUCX,MXSCGL5MQ,MXSCLG9PH,MXSCNPKJ,MXSCNPMT10,MXSCNPMT14,MXSCNPMT20,MXSCTGLMQ,MXSCTL24FK,MXSCTLWUC,MXSCVAGB50,MXSCVAGB51,MXSCVAW15E,MXSCVAW30E,MXSCVCLH15,MXSCVCLH20,MXSCVCLH30,MXSCVCLHB1,MXSDBSSMN,MXSDDCVTET,MXSDDCVTL,MXSDDCVTS,MXSDHRSPS,MXSDTMS421,MXSDTMS434,MXSDTMS435,MXSDTMS634,MXSEDCB,MXSEDG45D2,MXSEDG45D3,MXSEDG45D4,MXSEDG45T,MXSEDG4CV,MXSEDG60T,MXSEDGUC45,MXSEDGUC60,MXSEGIA60E,MXSEGIA60G,MXSEGIATUC,MXSEGMSTED,MXSEMSTBT,MXSERCPCW,MXSESNGAB,MXSESSIMT,MXSFLCVT1,MXSFLCVT3,MXSFLCVT4,MXSGIA80C,MXSGIAUC8D,MXSGIAUC8S,MXSGNLSZL,MXSGPLPSWG,MXSGPOIWG,MXSGTCPSI,MXSGTFTKB,MXSGTP2SI,MXSGTPEGKB,MXSGTRTBT,MXSGTSBT,MXSGTSTF20,MXSHDJWB,MXSHMPSM,MXSHRST33,MXSIJSTNB,MXSIPAMTM,MXSIPAMTN,MXSIPPMTM,MXSIPPMTN,MXSIVCFVTD,MXSIVSGPP,MXSIVSGPS,MXSJDJTMF,MXSJW450B,MXSLMBGPS,MXSLMEOPS,MXSLMEOPV,MXSLMESPS,MXSLMLSPS,MXSLMSGPS,MXSLMSPPS,MXSLNPT5LO,MXSLSIAHPH,MXSLSIHCPH,MXSLSP08PH,MXSLSP16PH,MXSLSS25PH,MXSLSS27PH,MXSMFG80D3,MXSMFG80D4,MXSMFG80S3,MXSMFG80S4,MXSMGGTVL,MXSMGTTPH,MXSNSFSTU,MXSNTECSDT,MXSNTECSFT,MXSNTECUST,MXSPAC58PH,MXSPAC78PH,MXSPCEEA2,MXSPCEEA25,MXSPCEEA28,MXSPCEEA31,MXSPCFBA2,MXSPCFBS10,MXSPCFBS70,MXSPCFBS71,MXSPCFBSK1,MXSPCFBSK7,MXSPCMPSI,MXSPCMSSI,MXSPCS201B,MXSPCSBSB,MXSPEEAXL,MXSPGHFJJ,MXSPGMLCVD,MXSPGMRCVD,MXSPGVPSM,MXSPMTA45,MXSPMTA60D,MXSPMTA60S,MXSPMTASTP,MXSPTGXBB,MXSPTNBSC,MXSPTNCBS,MXSPTSSPC,MXSPXBMTM,MXSPXBMTN,MXSRL100RT,MXSRL75TCR,MXSRL75TRT,MXSRLLN100,MXSRLLN75G,MXSRLLN75T,MXSRTCST,MXSRTCUCST,MXSRUECJJ,MXSSANRUG,MXSSETJMF,MXSSKRMVAT,MXSSKRMVPJ,MXSSKSTJJ,MXSSKSTPAT,MXSSKSTPRJ,MXSSKSTPWJ,MXSSMCSCT,MXSSPTZTTT,MXSSRMDTM,MXSSSTMTBL,MXSSTPCDH2,MXSSTPCDH3,MXSSTPCUC,MXSSVG30BB,MXSSVGB40B,MXSSVGIB4M,MXSTAUC45D,MXSTCR75UC,MXSTERCPCB,MXSTPZBKB,MXSTRCPTC1,MXSTRCPTC5,MXSTRFHTR,MXSTROCJJ,MXSTRT75UC,MXSTSDR25B,MXSTSDR50B,MXSTTWSI,MXSUPHSJJ,MXSUPM15JJ,MXSUPM6JJ,MXSURSTUG,MXSUTTXLB,MXSVAAP2LD,MXSVAFP3LD,MXSVCLAS4K,MXSVCLB45T,MXSVCLCL,MXSVCLGJF4,MXSVCLGJF5,MXSVCLGP40,MXSVCLGPS4,MXSVCLGST5,MXSVCLGST6,MXSVCLPSJ,MXSVCLS2,MXSVCVVUT,MXSVESP50B,MXSVESP51B,MXSVESP70B,MXSVESP71B,MXSVESR70B,MXSVGGT840,MXSVGGTS40,MXSVGHMS15,MXSVGHMS30,MXSVGHMS60,MXSVGPXT,MXSVGPXUC,MXSVGS25T,MXSVGS30T,MXSVGSP40T,MXSVGSP41T,MXSVGSPR4T,MXSVGTP40,MXSVGVVT,MXSVPTR20K,MXSVPTR80D,MXSVPTR80T,MXSVPTRIF6,MXSVPTRIF7,MXSVPTRIF8,MXSVSCL15,MXSVSP10B,MXSVSP12B,MXSVSP50BT,MXSVSP5B,MXSVST40K,MXSVTCVC7F,MXSVTEVHT,MXSWCBCTM,MXSWFFSBT,MXSWSC22B,MXSWSRBB,MXSXNSPNR,MXSXNSPPS,MXUBOSCS20,MXUBOSCSLV,MXUBOSFCMD,MXUBOSNDAS,MXUFD2636V,MXUFD8260,MXUHFGBLV,MXUHFGBNAL,MXUIVSTT,MXUNPMBLKB,MXUURMURTB,MXUUTSOBK,MXVAACMIT,MXVNCCMI,MXXACETAC,MXXAFXIST,MXXAMSOSJ,MXXANAASGT,MXXANCTRB,MXXANPSGIT,MXXAPDPBNH,MXXAPELIM,MXXAPESGWC,MXXAPILIM,MXXASLHHI,MXXASLLHI,MXXASSSHI,MXXATDSSJ,MXXBEGNVT,MXXBSCWHI,MXXBTCBCC,MXXBTCCA2,MXXCBDCJ,MXXCCDFMT,MXXCPSLPH,MXXCSGWLBN,MXXCSHGWBN,MXXDPTDVBB,MXXEBESGK,MXXEBSPFC,MXXECPTABI,MXXECPTBB,MXXEDASGBN,MXXEDASGIN,MXXELESGK,MXXEXCSBA,MXXEXCSBT,MXXFISLIM,MXXFTBMU,MXXFTPBMT,MXXGMTSAB,MXXGPPSAB,MXXGTTSGK,MXXGWPB,MXXHPSPFC,MXXICASTGA,MXXLVENBT,MXXLVENMF,MXXLVSTTD,MXXMBPTAB,MXXMBPTBT,MXXMBVVPT,MXXMCWACV,MXXMCWIDC,MXXMTDCB,MXXNAGNA2,MXXNAGNSCF,MXXNASGA2,MXXNASGWJ,MXXNAXDC1F,MXXNAXDC2F,MXXNAXDFA2,MXXNBKMVPH,MXXNBT8MPH,MXXNBTCBPH,MXXNBTCBUC,MXXNCP16B,MXXNCPBCPH,MXXNCPTABI,MXXNCPTWT,MXXNCRBCPH,MXXNCTEPB,MXXNECBCPH,MXXNECSLA2,MXXNECSLCB,MXXNEVGA2,MXXNEVGCJ,MXXNFBPCB,MXXNFGMGCP,MXXNFTMXCB,MXXNGBBLA2,MXXNGBBLPH,MXXNGDCCB,MXXNGDCCCB,MXXNGDDCCB,MXXNGSTGA2,MXXNGSTGCB,MXXNIDDA2,MXXNISDCFC,MXXNISDDFC,MXXNMCDPTD,MXXNMG2MPH,MXXNMG5MA2,MXXNMG5MPH,MXXNMG8MPH,MXXNMGBBA2,MXXNMGBBPH,MXXNMGSMA2,MXXNMHHCTD,MXXNMMP3TD,MXXNMRGWFC,MXXNMUPFTD,MXXNNTGWFC,MXXNNXTDCF,MXXNPPMA2,MXXNPPMCJ,MXXNPTSTFC,MXXNRBMCA2,MXXNRBMCFC,MXXNSBMGA2,MXXNSBMGWP,MXXNSCMGA2,MXXNSCMGWB,MXXNSFEPDF,MXXNSLTFC,MXXNSSGWFC,MXXNSTNRA2,MXXNSTNRDF,MXXNTGDA2,MXXNTGDCB,MXXNTMG10B,MXXNTWGHI,MXXNYSAAPH,MXXNZXDCA2,MXXPBFCAA,MXXPBFCAB,MXXPBFCIAB,MXXPBLCB,MXXPBLCIB,MXXPBSVL,MXXPBSVLI,MXXPBTCIB,MXXPBVFAB,MXXPBVPAB,MXXPBVPFSV,MXXPBVPSVI,MXXPCBCB,MXXPCFCAB,MXXPGDCB,MXXPGVCDA,MXXPLSTSBB,MXXPLUFHI,MXXPPBBLL,MXXPPBL018,MXXPPBLCBN,MXXPPBLCLI,MXXPPBLIDB,MXXPPBLLI,MXXPSBEB,MXXPSSESTB,MXXPSSLSEJ,MXXPSSXA2,MXXPSTALXP,MXXPSTBEBN,MXXPSTHLOL,MXXPSTPKBN,MXXPSTPTEF,MXXPSTSEBN,MXXPSTTIDB,MXXPTBLBT,MXXPTDCJ,MXXPTPBLBN,MXXPTSSBB,MXXPVSCWT,MXXPVSCWW,MXXPWSSA2,MXXPWSSEB,MXXPXTPBNH,MXXRBCMC,MXXRDCBBS,MXXRDCBTB,MXXRDDCBT,MXXRSGBM,MXXRWGCHI,MXXSFTKHI,MXXSJEC3S,MXXSJECM1,MXXSJECM2,MXXSJECM3,MXXSJECM4,MXXSJECMO,MXXSJES4C,MXXSMBSBL,MXXSMSGP2,MXXSMSGPM,MXXSMSSA2,MXXSMSSUPM,MXXSSLITM,MXXTASGBN,MXXTASGIN,MXXTFDA2S,MXXTFDSA2,MXXTFDSCJ,MXXTFDSRJ,MXXTFPCJS,MXXTPCPA2,MXXTPCPJS,MXXTREEVT,MXXTSPAA2,MXXVLCVPSG,MXXVLSGN,MXXY90GMS,MXXZAEVGBC,MXXZAPTCM,MXXZASGC,MXXZBEIBGC,MXXZCB2CM,MXXZCPTC,MXXZCVCBM,MXXZDSESTC,MXXZFADGC,MXXZFAPGC,MXXZFCDTC,MXXZHPCNC,MXXZIOPCM,MXXZIORCM,MXXZITDSC,MXXZITDTSC,MXXZPAGWC,MXXZPCNNC,MXXZPGDCC,MXXZPGDVOT,MXXZPGW18C,MXXZPGWC,MXXZPSGC,MXXZPSGEC,MXXZPTABC,MXXZPTCA2,MXXZRFIK07,MXXZRFIK12,MXXZRTVPC,MXXZRTVSC,MXXZTAAEGC,MXXZTXTAAC,MXXZVOTTC,MXZSZALCM,MYCE01,MYCE02,MYCI01,MYCL01,MYDE01,MYDT01,MYFT01,MYFT02,MYLT01,MYNT01,MYOI01,MYOT01,MYOT02,MYRT01,MYTE01,MYTE02,NACE01,NACH01,NALI01,NALI02,NANL01,NAPT01,NARI01,NASE01,NASE02,NASE03,NASI01,NAST01,NATE01,NATE02,NATT03,NAUI01,NAUT01,NAVI01,NAVI02,NAVT01,NAVT02,NAVT03,NEBI01,NEBT01,NED018,NED020,NED021,NED022,NED023,NED024,NED025,NED027,NELI01,NELI02,NELT01,NELT02,NELT03,NELT04,NEOE02,NEOE03,NEOE04,NEOI01,NEOL01,NEOL02,NEOT01,NEOT02,NESI01,NESI02,NESI03,NEST01,NETI01,NETI02,NEUE01,NEUE02,NEUI01,NEUI02,NEUI03,NEUI04,NEUI06,NEUT01,NEUT02,NEUT03,NEUT05,NEUT06,NEUT07,NEUT08,NEUT09,NEUT10,NEVE01,NEVT02,NEVT03,NEVT05,NEVT06,NEVT07,NEXI01,NEXT01,NEXT02,NIAT01,NIAT02,NIAT03,NICT01,NICT02,NICT03,NIDT01,NIFH01,NIFL01,NIFT02,NIFT03,NILT01,NILT02,NILT03,NIMI01,NIMI02,NIMI03,NIMT01,NIST01,NITE02,NITI01,NITI02,NITT01,NITT02,NIZE01,NIZEX1,NIZT01,NOLT01,NOLT02,NOOT01,NORI01,NORI02,NORI04,NORT01,NORT04,NORT06,NORT08,NORT09,NORT10,NORT11,NORT12,NORT13,NORT14,NORT15,NORT16,NORT17,NORT18,NORT19,NOV001,NOVE01,NOVI01,NOVI02,NOVI03,NOVI04,NOVI05,NOVT01,NOVT02,NOXL01,NOXT01,NSSI01,NSSI02,NSSI03,NSSI04,NSSI05,NSSI06,NSSI07,NSSI08,NSSI09,NSSI10,NSSI14,NSSIX1,NUBI01,NUET01,NUET02,NURT01,NURT02,NUTI01,NUTL01,NUTL02,NUTL03,NUTL04,NYOE01,OBIT01,OBIT02,OBUE01,OCII01,OCTI01,OCTI02,OCTI04,OCTI05,OCTI06,OESE01,OFLT01,OFLT02,OILE01,OKAE01,OLAT01,OLII01,OLII02,OLMT01,OLMT02,OLMT03,OMAT01,OMEH01,OMEL01,OMNE01,OMNI03,OMNI05,OMNL01,OMNT01,OMVI01,ONBI01,ONET01,ONET02,ONGT01,ONSI01,ONSI02,ONST01,ONST02,OPDI01,OPDI02,OPHI02,OPS001,OPS002,OPS003,OPS004,OPS005,OPS101,OPS102,OPS103,OPS104,OPTE01,OPTE02,ORAI01,ORAT01,ORBL01,ORF001,ORF002,ORF003,ORFT01,ORFT02,ORKT01,OROT01,ORSL02,ORT001,ORTT01,OSAT01,OSEH02,OSET02,OSET03,OSRL01,OSST01,OSST02,OSST03,OSTI01,OXAI01,OXAI02,OXAI03,OXI001,OXII01,OXII02,OXY001,OXY002,OXY003,OXYH01,OXYI02,OXYI03,OXYT01,OXYT02,OXYT03,OZUE01,PAMI01,PANE02,PANE04,PANE05,PANI01,PANI02,PANI03,PANL01,PANT01,PANT03,PARI01,PARI02,PARI03,PART02,PART03,PART04,PART05,PART06,PART07,PART08,PAST01,PATE01,PATE02,PAVI01,PAXI01,PAXI02,PCN001,PCN002,PEGI01,PEGI02,PEGI03,PEGI04,PEGI05,PEGI06,PEGI07,PEGI10,PEGI11,PEGI12,PEGLX1,PEME01,PENI01,PENI04,PENI05,PENI06,PENI07,PENI08,PENT01,PENT02,PENT04,PENT05,PENT06,PENT07,PEPL01,PEPL02,PER001,PER014,PERI02,PERI04,PERT01,PERT02,PERT03,PERT04,PETI01,PETI02,PGSI01,PGSI02,PH5E01,PH5E02,PHAE01,PHAI01,PHAT01,PHEE01,PHEH01,PHEH02,PHEI01,PHEI03,PHEL01,PHEL03,PHELX1,PHET01,PHET02,PHIT01,PHOLX1,PHOLX4,PHOLX5,PHOLX6,PINT01,PIRE01,PLAE01,PLAE02,PLALX1,PLAT01,PLAT02,PLAT03,PLAT04,PLAT05,PLAT06,PLAT07,PLET01,PLET02,PLET03,PLET04,PLET05,PLYE01,POCT01,PODE01,POL001,POL002,POL003,POLE01,POLE02,POLLX1,POLLX2,POLT01,POLT02,POLT03,POP001,PORE01,PORE02,PORT01,PORT02,POTC02,POTC04,POTC06,POTEX1,POTEX2,POTI01,POTLX5,POTLX6,POTL_1,POTL_2,POTL_3,POVEX1,POVEX3,POVEX4,POVEX6,PRAI01,PRAT01,PRAT02,PRAT03,PREE02,PREE03,PREE06,PREE07,PREEX1,PREH001,PREH002,PREI01,PREI02,PREI03,PREI04,PREI05,PREI06,PREI07,PREI08,PREL01,PREL02,PRET01,PRET02,PRET03,PRET05,PRET06,PRET08,PRET11,PRET12,PRET14,PRET15,PRET16,PRET17,PRET18,PRET19,PRET20,PRET22,PRET23,PRII01,PRII02,PRIT01,PRIT03,PRIT04,PRIT05,PRIT06,PRO001,PRO002,PROC02,PROE02,PROE03,PROE05,PROE06,PROH01,PROI01,PROI02,PROI03,PROI07,PROI08,PROI09,PROI10,PROI11,PROL02,PROT01,PROT02,PROT03,PROT06,PROT09,PROT10,PROT11,PROT12,PROT15,PROT16,PROT18,PROT20,PRUI01,PSET01,PSET02,PTD001,PTFE01,PTFE02,PTFE03,PTFE04,PTMF06,PTMF12,PTMF24,PTMF48,PTMF60,PTMF72,PTMM06,PTMM12,PTMM24,PTMM48,PTMM60,PTMM72,PULE01,PULE03,PURI01,PURI02,PURT01,PURT02,PURT03,PURT04,PYRH01,PYRT01,PYRT02,PYRT03,QBAC01,QCLE01,QUAT01,QUAT02,QUEL01,QUI002,QUI003,QUIE01,QUII01,QUIT02,QUIT03,QUOT01,QVAE01,RABI01,RABI04,RABT01,RABT02,RALT03,RANH01,RANH02,RANI01,RANI02,RANL01,RANL02,RANT01,RANT02,RAST01,REBI01,REBT01,REBT02,REBT03,RECI01,RECI02,RECI03,RECI04,RECI05,RECI06,RECI07,RED002,REDT01,REF001,REGE01,REGT01,RELE01,RELT01,REMI01,REMT01,REMT02,REMT03,REMT04,REMT05,REMT06,REMT07,RENI01,RENI02,RENT01,RENT02,RENT03,REOI01,REPE01,REPE02,REPT01,REQT01,REQT02,REQT03,RESE01,RESI03,RESI04,RESI05,RESL01,REST01,REST02,REST07,REVT01,REVT02,REVT04,REYT03,RHIE03,RHIL02,RIBI02,RICT01,RIDT01,RIFT01,RIFT02,RIFT03,RIFT04,RIFT05,RIFT06,RIFT07,RILT01,RILT03,RILT04,RIMH01,RIMT01,RINE01,RISI01,RISL01,RISL02,RIST01,RIST02,RIST03,RIST04,RIST05,RITI01,RITI02,RITI03,RITI04,RITI05,RITT01,RITT02,RITT03,RITT05,RITT06,RIVT01,RIVT02,ROAT01,ROAT02,ROBL01,ROCI02,ROCI04,ROCT01,ROML01,ROMT01,ROPT01,ROSI01,ROST01,ROST02,ROST03,ROST04,ROTI01,ROTT01,ROWT01,ROXT01,ROXT03,RULT01,RULT02,RULT03,RUPT01,RYTL01,RYTT01,SABT01,SAII01,SAII02,SALE01,SALE02,SALE03,SALEX2,SALEX4,SALEX6,SALEX7,SALEX9,SALL01,SALT01,SALT03,SALT04,SALT05,SALT06,SAML01,SAMT01,SAMT02,SAMT03,SAMT04,SANI01,SANI02,SANI03,SANI04,SANI05,SANI06,SANL01,SANT01,SANT02,SARL01,SAVEX1,SAVEX4,SAVEX5,SAVEX7,SAVEX8,SCA021,SCA023,SCA025,SCA027,SCAE01,SCHE01,SEBE01,SEBT01,SEEE01,SEFI01,SEL001,SEL002,SELT01,SEN001,SENT01,SENT02,SEPI01,SEPI02,SERE02,SERE03,SERE04,SERE05,SERE06,SERE07,SERE08,SERT02,SERT04,SERT05,SERT06,SERT07,SERT08,SERT09,SERT10,SERT11,SERT12,SERT13,SERT14,SERT15,SET001,SETO01,SETO02,SHOLX1,SHOLX2,SHOLX3,SHOLX4,SIAI01,SIAT01,SIAT02,SIAT03,SIAT04,SIAT05,SIAT06,SIAT07,SIBT01,SIFT01,SIFT02,SIFT03,SILE01,SILE02,SILEX2,SILEX3,SILH01,SILL01,SILT01,SIMI01,SIMI03,SIML02,SIMT01,SIN001,SINT01,SINT03,SINT04,SINT05,SINT06,SIPT01,SIPT02,SIRT01,SIXT01,SKI001,SKIEX1,SKIEX2,SKIEX3,SMEL01,SMOI01,SMOI02,SMOI03,SMWEX1,SMWEXS,SODC01,SODC04,SODC16,SODEX1,SODEX2,SODEX4,SODEX5,SODEX6,SODH01,SODI01,SODI02,SODI05,SODI06,SODLX1,SODT01,SODTX1,SODTX2,SOF001,SOFE01,SOFT04,SOFT05,SOL001,SOLE01,SOLE02,SOLI01,SOLI02,SOLI03,SOLI04,SOLI05,SOLI06,SOMI01,SOMI02,SOMI03,SOMI05,SOML01,SOML02,SOPT01,SOR001,SORC02,SORLX1,SORLX2,SORT01,SORT02,SOVT01,SPAT02,SPAT04,SPAT05,SPAT06,SPEE01,SPEE02,SPEE04,SPEI01,SPEL01,SPI022,SPI023,SPI025,SPIE01,SPIE02,SPIE03,SPIE04,SPIE05,SPIEX1,SPIH01,SPOI01,SPOL01,SPOT01,SPOT02,SPRT01,SPRT03,STAT01,STAT03,STAT04,STAT05,STAT09,STAT13,STAT14,STAT15,STER02,STER04,STET01,STET02,STIE01,STII01,STIT01,STIT02,STO102,STO104,STO612,STO614,STO616,STO618,STO903,STO904,STO905,STOE02,STOE03,STOLX3,STOT02,STOT03,STRI01,STRI02,STRI03,STRT03,STRT04,STRT05,STRT07,STRT08,STS001,STS002,STUT01,SUCL01,SUGT01,SULEX2,SULEX3,SULI01,SULI02,SULI03,SULI04,SULL01,SULT01,SULT02,SULT03,SUMT01,SUPT01,SUPT02,SUR001,SUR001_,SUR002,SUR003,SUR004,SUR005,SUR006,SUR007,SUR008,SUR009,SUR010,SUR020,SUR021,SUR101,SURI01,SUTT01,SUXI01,SWII02,SWIL01,SWIL02,SYI050,SYI100,SYME01,SYME02,SYME03,SYME04,SYNE01,SYNE04,SYNI01,SYNI02,SYNI03,SYNI05,SYNT01,SYR003,SYR006,SYR012,SYR020,SYR050,SYSE01,SYT025,TAFE01,TAFE02,TAGT06,TAMT01,TAMT02,TAMT03,TAMT04,TANT04,TAPE01,TAPT01,TARE01,TARL04,TART01,TART02,TART03,TART04,TART05,TART06,TAST01,TAST02,TAST04,TATI01,TAXI01,TAXI02,TAXI03,TAXI04,TAXI05,TAXI06,TAXI07,TAXI08,TAZI01,TDVI01,TEAE02,TEAI01,TEBT01,TECI01,TEET01,TEET02,TEET03,TEET04,TEET05,TEG001,TEG002,TEG003,TEG004,TEG005,TEG006,TEG007,TEG008,TEGL02,TEGT01,TEGT02,TELI01,TELI02,TELL01,TELT02,TELT03,TELT04,TEMT02,TEMT03,TEMT04,TEMT05,TEN002,TEN003,TEN004,TEN005,TEN006,TENT01,TENT02,TENT03,TENT04,TENT05,TENT06,TENT07,TENT08,TENT09,TENT10,TENT11,TENT12,TERE01,TERE02,TERE03,TERI01,TESI02,TETE01,TETE02,TETI01,TETI02,TETI03,TETI04,TETI05,TETT01,THAT01,THEI01,THIT01,THIT02,THIT03,THIT04,THR001,THRI01,THRT01,THYI01,THYI02,THYT01,THYT02,THYT03,TICT01,TICT02,TICT03,TIDT01,TIEI01,TIES01,TILI01,TIME02,TIME03,TIMT01,TOBE01,TOBE02,TOBE04,TOFL01,TOFT01,TOLT01,TOLT02,TONE01,TONE02,TONI02,TONT01,TOPE01,TOPE02,TOPE03,TOPI01,TOPI02,TOPT01,TOPT02,TOUI01,TPW001,TRA001,TRA101,TRA102,TRA103,TRA104,TRA105,TRA106,TRA107,TRA108,TRA109,TRAE01,TRAI01,TRAI02,TRAI03,TRAI04,TRAI06,TRAI07,TRAI08,TRAT01,TRAT02,TRAT04,TRAT05,TRAT06,TRAT07,TRAT08,TRAT10,TRCI01,TRCI02,TREI01,TRET02,TRI001,TRIE01,TRIE02,TRIEX1,TRIEX2,TRIEX3,TRIEX4,TRII01,TRIT01,TRIT07,TRIT08,TRIT11,TRIT12,TRIT13,TRIT14,TRIT15,TRIT16,TRIT17,TRO001,TRO002,TROC01,TROT01,TRUE01,TRUT01,TSOT01,TSOT02,TUBI01,TUBI02,TUMT01,TUSLX1,TWYT01,TYGI01,TYKT01,TYLT01,TYLT02,TYLT03,TYLT04,TYSL01,UCHT01,UFTT01,UFUT01,ULCL01,ULCL02,ULFT01,ULST01,ULST02,ULTE01,ULTT01,ULTT02,UNAI01,UNAT01,UNAT02,UNAT03,UNIE01,UNIE02,UNIE03,UNIT01,URAL01,URE001,UREE01,UREEX1,UREEX2,URET02,URI001,URI003,URI004,URI005,URIT02,URIT03,URIT04,URIT05,URO001,URO002,URO003,URO003_,URO004,UROI02,UROT01,UROT02,UROT03,URSH01,URSH02,URSH03,URST01,URST02,USPT01,UTMT01,UTRT01,UTRT02,VACI01,VACT01,VALI01,VALT01,VALT03,VALT04,VALT05,VALT06,VANH01,VANI01,VANI02,VANI03,VARI01,VARI02,VASI01,VAST02,VAXI01,VELI01,VELI02,VENE03,VENE04,VENE05,VENI01,VENI02,VENL01,VENT02,VEPI01,VERE01,VERI01,VERL01,VERT01,VERT02,VERT04,VERT05,VESI01,VESI02,VEST01,VEST02,VEST03,VEST04,VFEI01,VFEI02,VFEI03,VFET01,VFET02,VFET03,VIAT02,VICI01,VIDE01,VIDI01,VIDT01,VIDT03,VIGE01,VILE01,VILI01,VILT01,VILT02,VILT03,VILT04,VIMI01,VIMT01,VIMT02,VINI01,VINI02,VINI03,VINI04,VINI05,VINI06,VINI07,VIRC02,VIRT01,VISE01,VISI01,VISI02,VISI03,VISI04,VIST01,VITH01,VITH02,VITI01,VITI04,VITI05,VITI07,VITI08,VITI09,VITI10,VITI11,VITI12,VITLX1,VITLX2,VITT01,VITT03,VITT05,VITT07,VITT09,VITT10,VITT12,VITT13,VITT14,VIVE01,VIVT01,VIVT02,VOL001,VOLE01,VOLI01,VOLI02,VOLI03,VOLL01,VOLT02,VOLT03,VOLT04,VOLT05,VOPT01,VORH01,VOTT02,VULT01,VYTT01,VYTT02,WAL001,WAT001,WATE01,WATI02,WELT01,WHIC02,WHIEX1,WHIEX4,WHIEX5,WRI001,WRI002,WRI003,WRI004,XALE01,XALE02,XALT01,XAME01,XANE01,XANT01,XANT02,XANT03,XANT04,XANT06,XANT07,XANT08,XART02,XART03,XART04,XATT02,XELT01,XELT02,XENT01,XIGT01,XITT01,XUBL01,XYLE01,XYLE02,XYLI01,XYLI02,XYLI03,XYLI04,XYLI06,XYLIX4,XYLLX1,XYLLX2,XYZT01,YPSH00,ZADE01,ZADI01,ZALE01,ZALI01,ZANI01,ZANT01,ZAVI01,ZAVI02,ZAVT01,ZAVT02,ZBET01,ZEFI01,ZEFI02,ZEFI03,ZEFT01,ZELT01,ZELT02,ZEMT01,ZENI02,ZENL01,ZENL02,ZERT03,ZEST01,ZEST02,ZEVE01,ZEVI01,ZIAT02,ZIDT01,ZIDT02,ZIDT03,ZIDT05,ZIDT06,ZILT01,ZILT02,ZILT03,ZILT04,ZIMT01,ZIMT02,ZINC02,ZINEX1,ZINEX2,ZINI01,ZINI02,ZINL01,ZINLX1,ZINT01,ZINT02,ZITI01,ZITL01,ZITT02,ZMAL01,ZNSH01,ZOCT03,ZOFI01,ZOFI02,ZOFT01,ZOFT02,ZOLI01,ZOLI02,ZOLI03,ZOLT01,ZOLT02,ZOLT03,ZOMI01,ZONT01,ZOPT01,ZORL01,ZOVE02,ZOVI01,ZYKT01,ZYLT01,ZYLT02,ZYME01,ZYML01,ZYNE01,ZYNE02,ZYNE03,ZYPI01,ZYPT02,ZYRL01,ZYRT01,ZYRT02,ZYTT01,ZYVI01,ZYVT01 |
| icd10 | text | ICD-10 code (diagnosis) |

</p>
</details>

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
