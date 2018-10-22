
# Machine Learning Engineer Nanodegree
## Capstone Project
Guilherme Augusto Kater Marson  
September 26st, 2018

## I. Definition
This project will define a model to better evaluate the sell price of houses in Ames, Iowa. The data can be found [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). It뭩 important to mention that this dataset is part of a Kaggle competition, so it will be simple to compare the model performance to other model뭩 performance.  

### Project Overview
There are three values for any home on the market: What the seller thinks it뭩 worth, what the buyer thinks it뭩 worth and what a professional appraiser will think it뭩 worth. The seller wants as much money as possible for his house. The buyer wants to pay as low as possible and the professional appraiser will gather some data from the house, together with his background experience to come with a price. 
Seller and buyer usually trust the professional appraiser, but how accurate is his price? What if he has personal interest in completing this deal? Does he have all the data and knowledge to accurately evaluate that specific house? And the last point: isn뭪 he biased?


### Problem Statement
There are some ways to evaluate houses but the most used is the sales comparison method, that compares the house to be sold with similar properties in the same locality.
With this in mind and assuming that there is a database with explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, it뭩 possible to create a model to accurately evaluate house뭩 price. 
The model will try to regress the explanatory variables to the fairest price possible, removing from the equation the integrity and knowledge of the professional appraiser. 

### Metrics
The root-mean-square error (RMSE)  is a frequently used measure of the differences between values (sample or population values) predicted by a model or an estimator and the values observed. The RMSE represents the square root of the second sample moment of the differences between predicted values and observed values or the quadratic mean of these differences. 
RMSE is a measure of accuracy, to compare forecasting errors of different models for a particular dataset and not between datasets, as it is scale-dependent.
RMSE is always non-negative, and a value of 0 (almost never achieved in practice) would indicate a perfect fit to the data. In general, a lower RMSE is better than a higher one. However, comparisons across different types of data would be invalid because the measure is dependent on the scale of the numbers used.
RMSE is the square root of the average of squared errors. The effect of each error on RMSE is proportional to the size of the squared error; thus larger errors have a disproportionately large effect on RMSE. Consequently, RMSE is sensitive to outliers. 
The logarithm will be applied to both observed and predicted sales price so that the error predicting expensive and cheap houses will affect the result equally. 

## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
There are 2 datasets available to solve this problem. The first one is the train dataset and the second is the test dataset. The mais difference of them is that the test dataset does not have the target variables (SalePrice) because it's defined to be used as a submission to Kaggle website to evaluate the model. 
#### Train Dataset
 - Rows: 1460
 - Columns: 81
 - Data Types: 38 numerical columns and 43 textual columns 
 #### Test Dataset
  - Rows: 1459
 - Columns: 80
 - Data Types: 37 numerical columns and 43 textual columns 
#### Data  Sample
The dataset has more than 80 columns, so instead of showing all columns in this document, we will present only the first 10. 
![enter image description here](https://raw.githubusercontent.com/guilhermemarson/udacity/master/ames_house_prices/documentation/img/data_sample.JPG)

For a more detailed description on the variables, there is a file named variable_description.xlsx that holds the following information about each variable:

 - Name: the name of the variable
 - Type: numerical or categorical (textual)
 - Convert to Categorical: If it큦 a good idea to convert the numerical to a categorical variable
 - Description: detailed description about the variable
 - Expectation: my personal expectation about the variable. It will be used to guide the data exploration priorization



In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
#### Target Analysis
The target is a continuous variable, so the first step is to analyze its distribution:
![enter image description here](https://raw.githubusercontent.com/guilhermemarson/udacity/master/ames_house_prices/documentation/img/skewed_target_distribution.png)
As we can see, the distribution is right skewed. To illustrate a bit more, we will also analyze the Probability plot:
![enter image description here](https://raw.githubusercontent.com/guilhermemarson/udacity/master/ames_house_prices/documentation/img/skewed_target_probability_plot.png)
Usually predictive models work better on normally distributed targets. A log transformation on right skewed targets tends to be sufficient to normalyze it. Although we have already validated that there is no SalePrice that is 0 or negative, as a good practice we will use the ln(x+1) transformation.
Below we can see the target큦 distribution and probability plots after the log transformation: 
![enter image description here](https://raw.githubusercontent.com/guilhermemarson/udacity/master/ames_house_prices/documentation/img/normalized_target.png)
As we can see above, after the transformation the target is almost normal, so we will keep the log transformation on target and continue analysing the rest of the variables.
#### Numerical Variable Analysis
The first analysis on numerical variables is the Correlation Matrix, which brings the correlation between all numerical variables. We are going to use only the train dataset here because we want to take a closer look at how the variables correlates with the target (SalePrice)
![enter image description here](https://raw.githubusercontent.com/guilhermemarson/udacity/master/ames_house_prices/documentation/img/correlation_matrix.png)
Although we have some interesting correlations that needs to be investigated deeper, we will firstly focus on the variables that are highly correlated with SalesPrice.
* OverallQual
* TotalBsmtSF
* GrLivArea
* GarageCars
![Correlation plot of variables that are highly correlated with the target](https://raw.githubusercontent.com/guilhermemarson/udacity/master/ames_house_prices/documentation/img/target_highly_correlated.png)
Let's take a look at the scatter plots to have a better idea of the correlations between the variables and the target:
![Scatter plots](https://raw.githubusercontent.com/guilhermemarson/udacity/master/ames_house_prices/documentation/img/scatter_plots.png)
**OverallQual:** This numerical variable will be transformed to an ordered categorical variable. But  it큦 interesting to note that it has an almost linear correlation with the price
**TotalBsmtSF** This variable has a positive relationship with price. There is an interesting outlier that is a building with a very big basement, but I will keep it
**GrLivArea** and **GarageCars** are very correlated with the price as well with no outlier to be removed

From the above, only OverallQual and GarageCars are in my list of High Importance vars on "variable_description.xlsx" file. It큦 important to mention that I ranked the variables in the spreadsheet prior to analysing the data, so it's only a personal feeling on the variable importance.
The next step is to analyze variables that are highly correlated with itselves, not necessarily with the target. To avoid multicolinearity, the most correlated with the target will be kept and the other one will be dropped:
 - GarageArea Vs GarageCars
	 - GarageArea dropped
 - YearBuilt Vs GarageYrBlt
	 - GarageYrBlt dropped
 - GrLivArea Vs TotRmsAbvGrd
	 - TotRmsAbvGrd dropped
 - 1stFlrSF Vs TotalBsmtSF
	 - 1stFlrSF dropped

Before advancing to categorical variables, let큦 take a look at nulls
#### Null Variable Analysis
The null analysis is a very important step on data analysis. In this part we will find out if we have some variables with such a high amount of nulls that they can큧 even be used. In this step we will also define the rules for null imputation for all the variables. 
Although the act of using test data to analyze data and create null imputation rules can be considered data leakage, it's a common practice in Kaggle competitions in order to achieve better results. In real world modelling it would not be used, but for this specific case we are going to use.
 The graph below shows bars with the number of no null observations per variable. Each bar represents the number of not null observartion for each variable (higher bars are better). We will present only the variables that have at least 1 null observation:
 ![Nulls per variable](https://raw.githubusercontent.com/guilhermemarson/udacity/master/ames_house_prices/documentation/img/nulls%20per%20variable.png)
According to the data_description file, some nulls makes sense, so to avoid mistakes,  we will analyze them one by one, reading the variable's description file together with the null percentage.
* Alley : NA means No Alley access
* BsmtQual : NA means No basement
* BsmtCond : NA means No basement
* BsmtExposure: NA means No basement
* BsmtFinType1: NA means No basement
* BsmtFinType2 : NA means No basement
* BsmtExposure : NA means No basement
* BsmtFinType1 : NA means No basement
* BsmtFinType2 : NA means No basement
* FireplaceQu : NA means No fireplace
* GarageType : NA means No garage
* GarageFinish : NA means No garage
* GarageQual : NA means No garage
* GarageCond : NA means No garage
* PoolQC : NA means No pool
* Fence : NA means No fence
* MiscFeature : NA means None
* MasVnrType: NA means None
* Electrical: Mode fill
* KitchenQual: Mode fill
* MSZoning,SaleType: Mode fill 
* Exterior1st: Mode fill 
* Exterior2nd: Mode fill
* GarageYrBlt: Replacing missing data with 0 (Since No garage = no cars in such garage.)
* GarageArea and: Replacing missing data with 0 (Since No garage = no cars in such garage.) 
* GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
* MasVnrArea: NA means 0
* BsmtFinSF1: NA means 0
* BsmtFinSF2: NA means 0
* BsmtUnfSF: NA means 0
* TotalBsmtSF: NA means 0
* BsmtFullBath: NA means 0
* BsmtHalfBath: NA means 0
* Functional : NA means typical
* LotFrontage : Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
* LGarageYrBlt :  All garages that do not have a Year Built, does not exist (has 0 size in cars), so we will fill nulls with 0
#### Categorical Variables Analysis
The objective of the catagorical variable analysis is to identify variables that can explain the target. Also, it큦 very important to identify if the data is not concentrated on only one category or if the categorical has only one category. 
Since the number of categorical variables is too high and all of the analysis is in the data_exploration.ypnb file, we will cover only the most important findings in this section

![enter image description here](https://raw.githubusercontent.com/guilhermemarson/udacity/master/ames_house_prices/documentation/img/Alley.png)
![enter image description here](https://raw.githubusercontent.com/guilhermemarson/udacity/master/ames_house_prices/documentation/img/CentralAir.png)
![enter image description here](https://raw.githubusercontent.com/guilhermemarson/udacity/master/ames_house_prices/documentation/img/HEatingQC.png)
![enter image description here](https://raw.githubusercontent.com/guilhermemarson/udacity/master/ames_house_prices/documentation/img/KitchenQual.png)
![enter image description here](https://raw.githubusercontent.com/guilhermemarson/udacity/master/ames_house_prices/documentation/img/MSZoning.png)
![enter image description here](https://raw.githubusercontent.com/guilhermemarson/udacity/master/ames_house_prices/documentation/img/Neighborhood.png)
![enter image description here](https://raw.githubusercontent.com/guilhermemarson/udacity/master/ames_house_prices/documentation/img/Utilities.png)


In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model뭩 solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model뭩 final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you뭭e written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
