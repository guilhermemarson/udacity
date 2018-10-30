##  Project Summary
This project will define a model to better evaluate the sell price of houses in Ames, Iowa. The data can be found "src" folder. 
## Folders Organization
 - Root - Ipython Notebooks files
	 - data_exploration.ipynb - this file contains all data exploration codes
	 - data_pipeline.ipynb - as a result of the data exploration, this file was created to perform all data pre-processing steps
	 - tpot_benchmark.ipynb - This file has the code of the automated machine learning code used as the first benchmark without going through any data pre-processing
	 - model.ippynb - Code of the first 2 models created (Random Forest and Gradient Boosting)
	 - tpot_after_data_pipeline.ipynb - Tpot Automated machine learning model created after data pre-processing
	 - model_validation.ipynb - code created to investigate overfitting and to create a fourth model (Gradient Boosting) avoiding overfitting
 - src folder
	 - train and test csv files
 - output folder
	 - all model output files
 - documentation folder
	 - img folder
		 - all images used on the report file
	 - Hose Evaluation.docx - source of proposal.pdf
	 - proposal.pdf - project proposal
	 - references.txt - project references
	 - report.md - Final Project Report source
	 - **report.pdf** - Final Project Report
	 - variable_description - description of all the variables used to create the model
## Libraries
 - pandas for data manipulation
 - numpy for scientifc computing
 - scipy for some statistics
 - matplotlib and seaborn for plots
 - missingno for null analysis
 - scikit learn for machine learning
 - tpot for automated machine learning
