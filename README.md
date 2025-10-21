## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
      
import numpy as np

from scipy import stats

df=pd.read_csv("data.csv")

df

<img width="724" height="541" alt="image" src="https://github.com/user-attachments/assets/8f41f4a9-7983-4b1a-96b1-5a18353ae02a" />


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

climate=['Cold','Warm','Hot','Very Hot']

ele=OrdinalEncoder(categories=[climate])

ele.fit_transform(df[["Ord_1"]])



<img width="195" height="291" alt="image" src="https://github.com/user-attachments/assets/1434c4f5-dc95-40c9-b76e-065043c80759" />


df['bo2']=ele.fit_transform(df[['Ord_1']])

df


<img width="758" height="542" alt="image" src="https://github.com/user-attachments/assets/1957a03f-a176-48cc-b9d1-7c5d3839c75e" />


le=LabelEncoder()

df2=df.copy()

df2['Ord_2']=le.fit_transform(df2['Ord_2'])

df2

<img width="710" height="536" alt="image" src="https://github.com/user-attachments/assets/78d95125-7651-40ae-9e3c-fc140596e617" />


from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder()

df3=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df[['City']]))

df2=pd.concat([enc,df3],axis=1)

df2


<img width="819" height="333" alt="image" src="https://github.com/user-attachments/assets/3d8b2343-2d3c-48cd-9264-8387a131879b" />


pd.get_dummies(df,columns=['City'])


<img width="817" height="349" alt="image" src="https://github.com/user-attachments/assets/ec923969-c7bb-4ccd-a8bf-aba61cbdda46" />


pip install --upgrade category_encoders


<img width="815" height="189" alt="image" src="https://github.com/user-attachments/assets/d64d80d1-428d-44f4-98ff-fd925ea828a5" />


from category_encoders import BinaryEncoder

import pandas as pd

df=pd.read_csv("C:\Users\priya\Downloads\data.csv")

be=BinaryEncoder()

nd=be.fit_transform(df['Ord_2'])

df1=pd.concat([df,nd],axis=1)

df1=df.copy()

df1


<img width="807" height="562" alt="image" src="https://github.com/user-attachments/assets/f1e973d6-3685-475d-8247-78f3b4c3e93c" />


from category_encoders import TargetEncoder

te=TargetEncoder()

cc=df.copy()

new=te.fit_transform(X=cc["City"],y=cc["Target"])

cc=pd.concat([cc,new],axis=1)

cc


<img width="815" height="506" alt="image" src="https://github.com/user-attachments/assets/09a7ca03-38fe-4677-b620-5180d252e293" />


import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df


<img width="817" height="423" alt="image" src="https://github.com/user-attachments/assets/cc6bf59b-76e1-4ce5-895f-6d9d7c0f3c85" />


df.skew()


<img width="489" height="162" alt="image" src="https://github.com/user-attachments/assets/19d56d12-c431-4f32-98c1-d1aee741fd4d" />


np.log(df["Highly Positive Skew"])


<img width="810" height="376" alt="image" src="https://github.com/user-attachments/assets/64506679-03e3-4f45-a8c7-f46ddb842985" />


np.reciprocal(df["Highly Positive Skew"])


<img width="816" height="332" alt="image" src="https://github.com/user-attachments/assets/893ebcd6-20a4-4995-9420-2f9033ffeb5f" />

np.reciprocal(df["Moderate Positive Skew"])

<img width="818" height="371" alt="image" src="https://github.com/user-attachments/assets/a809c262-e658-46cd-8a05-b909d4463268" />

np.square(df["Highly Positive Skew"])


<img width="808" height="355" alt="image" src="https://github.com/user-attachments/assets/49f0012c-a230-4503-8670-326c00ddbeac" />


df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])

df

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df['Moderate Negative Skew'])

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])

df



<img width="816" height="283" alt="image" src="https://github.com/user-attachments/assets/c51dfc78-4fae-477a-b1f6-aee959124298" />

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

import scipy.stats as stats

sm.qqplot(df['Moderate Negative Skew'],line='45')

plt.show()


<img width="821" height="486" alt="image" src="https://github.com/user-attachments/assets/5ba995af-f8cd-45d0-b118-6b7499ce2800" />


sm.qqplot(df['Moderate Negative Skew_1'],line='45')


<img width="819" height="463" alt="image" src="https://github.com/user-attachments/assets/ddd33b94-0745-4e32-ae12-4d3376cf0a09" />


df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])

sm.qqplot(df['Highly Negative Skew'],line='45')

plt.show()


<img width="817" height="490" alt="image" src="https://github.com/user-attachments/assets/b8fe9528-eea2-4bec-8c42-94177dde1d38" />


sm.qqplot(df['Highly Negative Skew_1'],line='45')

plt.show()


<img width="815" height="452" alt="image" src="https://github.com/user-attachments/assets/bbddb48d-4256-498d-b0cc-b2e48755b6a7" />


sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')


<img width="816" height="505" alt="image" src="https://github.com/user-attachments/assets/15beb262-03be-4106-8bcb-0ad1a4a38aaf" />

# RESULT:
       Thus the given data,Feature Encoding,Transformation process and save the data to a file was performed successfully.
       
