# Use machine learning model to predict ICD-10

## Team member
1. Assistant Professor Piyapong Khumrin, MD, Faculty of Medicine, Chiang Mai University, Chiang Mai, Thailand
2. Dr. Krit Khwanngern, MD, Faculty of Medicine, Chiang Mai University, Chiang Mai, Thailand
3. Associate Professor Nipon Theera-Umpon, PhD, Biomedical Engineering Institute, Chiang Mai University
4. Terence Siganakis, CEO, Growing Data Pty Ltd
5. Alexander Dokumentov, Data Scientist, Growing Data Pty Ltd

## Duration
6 months (March - August 2019)

## Introduction
Over one million patient visit Maharaj Nakhon Chiang Mai hospital at the outer patient department (reported in [2018](http://www.med.cmu.ac.th/hospital/medrec/2011/index.php?option=com_content&view=category&id=130&Itemid=589)). Every year, the hospital needs to report data to the government for billing. 
### Problem statement
The amount of budget which can be claimed from the billing depends on the quality and completeness of the document. One major problem is the completeness of diagnosis (representing in ICD-10 code). The current process to complete the diagnosis in the document is a labour intersive work which requires physicians or technical coders to review medical records and manually enter a proper diagnosis code. Therefore, we see a potential benefit of machine learning application to automate this ICD-10 labelling process.
### Prior work
[ICD-10](https://en.wikipedia.org/wiki/ICD-10) is a medical classification list for medical related terms such as diseases, signs and symptoms, abnormal findings, defined by the World Health Organization (WHO). In this case, ICD-10 is used to standardized the diagnosis in the billing report before submitting the report to the government. Prior research showed the success of applying machine learning for auto-mapping ICD-10. 

[Serguei et al.](https://watermark.silverchair.com/13-5-516.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAmkwggJlBgkqhkiG9w0BBwagggJWMIICUgIBADCCAksGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM_rB6l2CXCEn65rFzAgEQgIICHNR0jGPFkz5CcmYleFTpxEzR1savr-C4O6Z00kG0BdtRIJS3RYpvI82St6UFm3i1S9e3JiFyMnm9LvsD3t0KiqIrBnruV91HJiba-iLVcANAs2U2P9bSpD4MiJizcuRH4pW7v6E2b5uEIwQ5j0QGrjrvSsRCeuQA3kFSPFCOMxq7HBBLMDLsFlXSBERfHxtnkMhRBtrxRQZ3a_tAgJ70oRFF5LiTJpbANSmhGShufU3H6HlhPJSxetkkznEaZ_1r4e26TmnFYHvNjAUwLDSrV3eopiKjxnR_s-mDe6PWRoeXWmBz2JmB29l0IE9fkG86wlgx-7J0FGuJjOMbRBDTvo2TzU2cPqwcUNlBCcNgN2P2o2PY8rjw_Gr4J-DmuLNjkxayrSYZVi3p75YGaQr_XLwPw-HkTqZtdjwCk9fA2GzcNubW4l4k2o4469PImjxmDnm6gXi_STkQ0im_dc6eXZP8Z_fHiEI4gxWA4I4Br_h021L29TgRiva-thtvaPROop3G3lw1HtMy5mArx_16IfTHcFGXF28N3AH2qFert6cfPnbOZn-FGqKUQVJCce6Cfz8FLjpWq2qYZ2CeWMUC3IjooCHp7VtjGxAW4il61LD2stSJSrv4OMBeWyciPNTJczrZOUVyjX8MIVupQvFVPwOBWW3Q7a8N4mZhI3XuYlZanI0gvfp3KXbtg6NMJxBrAD3yztwGkslU1zLiTw) applied simple filters or criteria to predict ICD-10 code such as assuming that a high frequency code that shares the same keywords is a correct code, or gender specific diagnsis (use gender to separate female and male specific diseases). Other cases which were not solved by those filters, then use a machine learning model (Naive Bayes) or bag of words technique. They used the SNoW implementation of the naïve Bayes classifier to solve the large number of classification, and Phrase Chunker Component for mixed approach to solve a classification. The model evaluation showed that over 80% of entities were correctly classified (with precision, recall, and F-measure above 90%). The limitations of the study were the features were treated independently which might lead to a wrong classification, continuous value such as age interfered how the diseases are classified, and lower number of reference standard (cased manually labelled by human coder).

[Koopman et al.](https://www.sciencedirect.com/science/article/pii/S1386505615300289) developed a machine learning model to automatically classify ICD-10 of cancers from free-text death certificates. Natural language processing and SNOMED-CT were used to extract features to term-based and concept-based features. SVM was trained and deployed into two levels: 1) cancer/nocancer (F-measure 0.94) and 2) if cancer, then classify type of cancer (F-measure 0.7). 

[Medori and Fairon](https://aclanthology.info/pdf/W/W10/W10-1113.pdf) mapped clinical text with standardized medical terminologies (UMLS) to formulate features to train a Naive Bayes model to predict ICD-6(81% recall). 

[Boytcheva](http://www.aclweb.org/anthology/W11-4203) matched ICD-10 codes to diagnoses extracted from discharge letters using SVM. The precision, recall, F-measure of the model were 97.3% 74.68% 84.5%, respectively. 

In summary, prior research shows that machine learning model plays a significant and beneficial role in auto-mapping ICD-10 to clinical data. The common approach of the preprocessing process is using NLP process to digest raw text and map with standardized medical terminologies to build input features. This is the first step challenge of our research to develop a preprocessing protocol. Then, the second step is to design an approach how to deal with a large number of input features and target classes (ICD-10).

Our objectives are to develop machine learning model to mapp missing or verify ICD-10 in order to obtain more complete billing document. We aim to test if we use the model to complete ICD-10 for one year report and evaluate how much more the hospital can claim the billing. 

## Objectives
1. Use machine learning models to predict missing ICD-10.
2. Use machine learning models to verify ICD-10 labelled by human.

## Aims
1. The performance of machine learning model shows precision, recall, and F-measure are greater than 80%.
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
  
## How to use
## How it works
## Model evaluation
## Limitations
