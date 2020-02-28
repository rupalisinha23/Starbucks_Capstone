# Starbucks Capstone Project
### This project is the part of Udacity's Machine Learning Engineering Nanodegree program.

## Problem Statement
Starbucks collects the customer data to understand their behaviour on the rewards and offers sent via the mobile-app. Once every few days, Starbucks sends the personalised offers to its customers. These customers can respond positively/negatively/neutrally. A key thing to note is that not all the customers receive the same offer. The task of this project is to combine transaction, demographic and offer data of the past (which is already provided) to determine which demographic groups respond best to which offer types. 


## Python Dependencies
:pushpin: numpy <br>
:pushpin: pandas <br>
:pushpin: math <br>
:pushpin: json <br>
:pushpin: matplotlib <br>
:pushpin: seaborn <br>
:pushpin: os <br>
:pushpin: sklearn <br>
:pushpin: xgboost <br>
:pushpin: sagemaker <br>
:pushpin: boto3 <br>

## AWS dependencies
:door: Make sure to have an aws account. Otherwise please go to https://aws.amazon.com/console/. <br>
:door: Make sure to have IAM. Otherwise refer to https://docs.aws.amazon.com/directoryservice/latest/admin-guide/setting_up_create_iam_user.html. <br>
:door: Make sure to have a sagemaker role. Otherwise refer to https://docs.aws.amazon.com/glue/latest/dg/create-an-iam-role-sagemaker-notebook.html. <br>

<b>NOTE: </b>Please follow the instructions for setting aws as given in the nanodegree course or refer to the above mentioned links.

 
## Getting Started:
#### Installing Python dependencies
- Clone or download the project as it is.
- Go to the cloned repo via Terminal/CommandPrompt. Execute below commands to create and activate virtual environment.
```
$ python3 -m venv virtual-env-name
$ source virtual-env-name/bin/activate
$ pip install -r requirements.txt
$ pip install ipython ipykernel jupyterlab
$ jupyter notebook
```

Once the jupyter console gets opened, open the terminal and install aws cli. Refer to https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html.
Next, we need to configure it. Write:
```
aws configure
```
To configure aws, refer to https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html


## Conclusions
The problem addressed in this project was that we wanted to personalise sending the different kinds of offers to the customers because not all the offers should be sent to all the customers randomly. Hence, we turned this problem into a supervised learning task where we trained a classifier given the features related to the customer’s profile, offer’s portfolio and customer’s transaction data, predict whether an offer will be converted/successful or not. To do this, we first implemented a baseline of logistic regression because that is the first and foremost algorithm which comes to the mind if the target data is binary (categorical) in nature. Logistic regression gives the probability of a sample belonging to both the classes. We found that the model performance was good with an average performance of 85% taking all the four metrics together. However, we then tried two advanced algorithms. First we implemented Linear-SVC and found the performance of the model to be around 87% taking all the metrics together. Then we implemented XGBoost to have a performance jump to 90% taking all the metrics together. This is also because XGBoost is quite an efficient algorithm and known to outclass all other classifiers. It is capable of performing the three main forms of gradient boosting (Gradient Boosting (GB), Stochastic GB and Regularized GB) and it is robust enough to support fine tuning and addition of regularization parameters. System-wise, the library’s portability and flexibility allow the use of a wide variety of computing environments like parallelization for tree construction across several CPU cores; distributed computing for large models and Cache Optimization to improve hardware usage and efficiency. With all these advantages, we still face a miss-classification error of around 8%.


### Author
* **Rupali Sinha** - *Initial work*
