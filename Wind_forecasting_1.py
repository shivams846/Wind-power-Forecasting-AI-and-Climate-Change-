#!/usr/bin/env python
# coding: utf-8

# # Wind Power Forecasting: Design Phase (Part 1)
# 
# 
# In the previous lab, you performed an exploratory analysis of the [SDWPF dataset](https://arxiv.org/abs/2208.04360), which contains data from 134 wind turbines from a wind farm in China. The SDWPF data was provided by the Longyuan Power Group, which is the largest wind power producer in China and Asia.
# 
# In this lab you will begin to design a solution for wind power forecasting. The steps you will complete in this lab are:
# 
# 1. Import Python packages
# 2. Load the dataset
# 3. Catalog abnormal values
# 4. Establish a baseline for wind energy estimation
# 5. Perform feature engineering \
#     5.1 Delete redundant features - Pab \
#     5.2 Transform angle features \
#     5.3 Fix temperatures and active power features \
#     5.4 Create time features
# 6. Update linear model baseline with more features
# 7. Use a neural network to improve wind power estimation

# ## 1. Import Python packages
# 
# Run the next cell to import that Python packages you'll need for this lab.
# 
# Note the `import utils` line. This line imports the functions that were specifically written for this lab. If you want to look at what these functions are, go to `File -> Open...` and open the `utils.py` file to have a look.

# In[ ]:


import numpy as np # package for numerical calculations
import pandas as pd # package for reading in and manipulating data
import utils # utility functions for this lab

print('All packages imported successfully!')


# ## 2. Load the dataset
# 
# 
# The original dataset contains information of 134 turbines, and when you run the next cell you will read in the data, then perform the same steps you ran in the last lab, namely, select the top 10 turbines that produced the most power on average, and convert the day and timestamp columns into a single datetime column.

# In[ ]:


# Load the data from the csv file
raw_data = pd.read_csv("./data/wtbdata_245days.csv")

# Select only the top 10 turbines
top_turbines = utils.top_n_turbines(raw_data, 10)

# Format datetime (this takes around 15 secs)
top_turbines = utils.format_datetime(top_turbines, initial_date_str="01 05 2020")

# Print out the first few lines of data
top_turbines.head()


# ## 3. Catalog abnormal values

# If you read the paper associated with this dataset you will see a section called `Caveats about the data`, which mentions that some values should be excluded from the analysis because they are either `missing`, `unknown` or `abnormal`. 
# 
# `missing` values are self explanatory but here are the definitions for the other two types:
# 
# `unknown`:
# - if `Patv` ≤ 0 and `Wspd` > 2.5
# - if `Pab1` > 89° or `Pab2` > 89° or `Pab3` > 89° 
# 
# `abnormal`:
# - if `Ndir` < -720 or `Ndir` > 720
# - if `Wdir` < -180 or `Wdir` > 180
# 
# When you run the next cell you will create a new column called `Include` in the dataframe and set the value to False for every `missing / unknown / abnormal` value:

# In[ ]:


# Initially include all rows
top_turbines["Include"] = True

# Define conditions for abnormality
conditions = [
    np.isnan(top_turbines.Patv),
    (top_turbines.Pab1 > 89) | (top_turbines.Pab2 > 89) | (top_turbines.Pab3 > 89),
    (top_turbines.Ndir < -720) | (top_turbines.Ndir > 720),
    (top_turbines.Wdir < -180) | (top_turbines.Wdir > 180),
    (top_turbines.Patv <= 0) & (top_turbines.Wspd > 2.5)
]

# Exclude abnormal features
for condition in conditions:
    top_turbines = utils.tag_abnormal_values(top_turbines, condition)
    
top_turbines.head()


# Now run the next cell to create the `clean_data` dataframe which no longer includes all data since abnormal values have been removed:

# In[ ]:


# Cut out all abnormal values
clean_data = top_turbines[top_turbines.Include].drop(["Include"], axis=1)

clean_data.head()


# ## 4. Establish a baseline for wind power estimation
# 
# Before moving forward you will create a baseline for wind power estimation using a `linear regression` model to fit the relationship between wind speed and power output.  
# 
# You can use the dropdown to train a linear model for of of the turbines and see how it performs by looking at a plot of predicted vs actual power output values and mean absolute error for the model. 

# In[ ]:


utils.linear_univariate_model(clean_data)


# ## 5. Feature engineering

# Before building a model capable of estimating power output from the other features you need to perform some `Feature Engineering`. During this process you will transform your existing features into better representations, combine features, fix issues with them and create new features.

# ### 5.1  Delete redundant features - Pab
# 
# In the previous lab you saw that all the `Pab#` features (which stands for `pitch angle blade #`) were perfectly correlated, which means that they are redundant. You can instead keep only one of these features and rename it as `Pab`. Run the next cell to keep only 1 column of `Pab` features.

# In[ ]:


# Aggregate pab features
clean_data = utils.cut_pab_features(clean_data)

clean_data.head(5)


# ### 5.2 Transform angle features
# 
# There are 3 features (`Wdir`, `Ndir`, `Pab`) which are encoded in degrees. This is problematic because your model has no way of knowing that angles with very different values (such as 0° and 360°) are actually very similar (the same in this case) to each other. To address this you can transform these features into their `sine`/`cosine` representations. 
# 
# Run the next cell to convert angle features to their `sine`/`cosine` representations.

# In[ ]:


# Transform all angle-encoded features
for feature in ["Wdir", "Ndir", "Pab"]:
    utils.transform_angles(clean_data, feature)  
    
clean_data.head(5)


# ### 5.3 Fix temperatures and active power
# 
# You might remember from the previous lab that both `Etmp` and `Itmp` had really negative values. In fact, these minimum values are very close to the absolute zero (-273.15 °C) which is most certainly an error. Here you will use linear interpolation to fix these values.
# 
# Active power has negative values which doesn't make sense in the context of the problem at hand. The paper also addresses this issue by mentioning that all negative values should be treated as zero. 
# 
# You can apply these changes by running the following cell:

# In[ ]:


# Fix temperature values
clean_data = utils.fix_temperatures(clean_data)

# Fix negative active powers
clean_data["Patv"] = clean_data["Patv"].apply(lambda x: max(0, x))

clean_data.head(5)


# ### 5.4 Create time features
# 
# You will create features that encode the time-of-day signals for each data point in the dataset. 
# 
# If you curious about how this encoding works be sure to check out this [post](https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/). 

# In[ ]:


# Generate time signals
clean_data = utils.generate_time_signals(clean_data)

clean_data.head(5)


# Run the next cell to do a final step to prepare your data for modeling.

# In[ ]:


# Define predictor features 
predictors = [f for f in clean_data.columns if f not in ["Datetime", "TurbID", "Patv"]]

# Define target feature
target = ["Patv"]

# Re-arrange features before feeding into models
model_data = clean_data[["TurbID"]+predictors+target]

model_data.head(5)


# ## 6. Update linear model baseline with more features
# 
# Now that you have performed some feature engineering phase it's time to try some more modeling with your new set of  features. You can use the dropdown to select the turbine to model and choose from the list of features you want to include in the model. Use the shift and arrow keys on your keyboard to select the features you wish to include and then click on the `Run Interact` button to train your model.
# 
# Notice that since you are including more features it is not possible to visualize the fitted model in 2 dimensions. With this in mind, that plot is replaced by one that shows the average feature importance for every feature you include:

# In[ ]:


# Create a linear model with more features
utils.linear_multivariate_model(model_data, predictors)
# Running the interaction below might take a minute


# ## 7. Use a neural network to improve wind power estimation
# 
# Now you will train a neural network model for comparison. As in the previous section you can use the dropdown to select the turbine to model and choose features you want to include from the list. Click on the `Run Interact` button to train the network and output the results.

# In[ ]:


# Train a neural network model
utils.neural_network(model_data, predictors)
# Running the interaction below might take a minute


# ## **Congratulations on finishing this lab!**
# 
# **Keep up the good work :)**

# In[ ]:




