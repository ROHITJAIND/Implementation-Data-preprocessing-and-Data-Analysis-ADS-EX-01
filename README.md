# EX-01 Implementation of Data Preprocessing and Data Analysis
### Aim:
To implement Data analysis and data preprocessing using a dataset.&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**DATE: 08.09.2024**
<br>

### Algorithm:
- Step 1: Import the dataset necessary.
- Step 2: Perform Data Cleaning process by analyzing sum of Null values in each column a dataset.
- Step 3: Perform Categorical data analysis.
- Step 4: Use Sklearn tool from python to perform data preprocessing such as encoding and scaling.
- Step 5: Implement Quantile transfomer to make the column value more normalized.
- Step 6: Analyzing the dataset using visualizing tools form matplot library or seaborn.
<br>

### Program:
##### Importing required python libraries
```Python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
```
<br>

##### Importing and Analyzing the Dataset
```Python
df=pd.read_csv("Toyota.csv")
df.head(10)
df.isnull().sum()
df.info()
```
<br>

**df.head(10)**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**df.isnull().sum()**&emsp;&emsp;&emsp;**df.info()**<br>
<img width=50% valign=top height=17% src="https://github.com/user-attachments/assets/2d9f138f-b58e-45cc-9573-0276cd353cad">
<img width=20% valign=top height=17% src="https://github.com/user-attachments/assets/f7107318-31a0-43d0-b830-9be5dc8a3751">
<img width=28% valign=top height=17% src="https://github.com/user-attachments/assets/c14cd2bf-9305-44ad-bbbd-267d70e74e34">
<br>

##### Preprocessing the Data
```Python
df=df.drop(df[df['KM'] == '??'].index)
df=df.drop(df[df['HP']=='????'].index)
df=df.drop('Unnamed: 0',axis=1)
df['Doors']=df['Doors'].replace({'three':3,'four':4,'five':5}).astype(int)
df[['FuelType','MetColor']]=df[['FuelType','MetColor']].fillna(method='ffill')
df[['Age']]=df[['Age']].fillna(df['Age'].mean()).astype(int)
df[['KM','HP']]=df[['KM','HP']].astype(int)
```
<br>

##### Detecting and Removing Outliers:
```Python
numeric= ['Price','Age','KM','HP','CC','Automatic','Weight']
plt.figure(figsize=(8, 3 * ((len(numeric)) // 3)))
for i, column in enumerate(numeric, 1):
    plt.subplot((len(numeric)+3 ) // 3, 3, i)
    sns.boxplot(x=df[column])
    plt.title(f'{column}')
plt.tight_layout()
plt.show()
for column in numeric:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
plt.figure(figsize=(8, 2 * ((len(numeric)) // 3)))
for i, column in enumerate(numeric, 1):
    plt.subplot((len(numeric)) // 3,3, i)
    sns.boxplot(x=df[column])
    plt.title(f'{column}')
plt.tight_layout()
plt.show()
```

<table>
<tr>
<td width=50%>
  
**Before Removing Outliers**  
![download](https://github.com/user-attachments/assets/8d1c379f-5b03-44b6-a9da-557fa704cc18)

</td> 
<td>
  
**After Removing Outliers** 
![download](https://github.com/user-attachments/assets/94bd3ffa-e5fd-4bb2-8bcb-6f8726a2347b)


</td>
</tr> 
</table>
<br>

##### Identifying Categorical data and performing Categorical analysis

<table>
<tr>
<td width=30%>
  
```Python
category=df.select_dtypes(include=['object'])
count=category.value_counts()
count.plot(kind='bar', color='yellow')
plt.title('Count of Fuel Types')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='x', linestyle='solid', alpha=0.7)
plt.show()
```
</td> 
<td>
<img src="https://github.com/user-attachments/assets/9d5462ec-1d67-4b5e-aebc-9f45b3a0c734">
</td>
</tr> 
</table>

##### Performing Bivariate Analysis


<table>
<tr>
<td width=40%>
  
```Python
sns.lineplot(x=df['Age'], y=df['Price'])
plt.title('Bivariate Analysis: Price vs Age')
plt.xlabel('Age')
plt.ylabel('Price')
plt.grid(True)
plt.show()
```
</td> 
<td>

![download](https://github.com/user-attachments/assets/c4b0dec9-e917-4376-9ba5-52311b1303da)
</td>
</tr> 
</table>
<br>

##### Performing Multivariate Analysis

<table>
<tr>
<td width=40%>
  
```Python
sns.countplot(x='HP', hue='MetColor', data=df)
plt.title('Count Plot: HorsePower and MetalColor')
plt.grid(True)
plt.show()
```

</td> 
<td>

![download](https://github.com/user-attachments/assets/5e98d0e5-62b3-4504-8754-0760175ac07f)
</td>
</tr> 
</table>

##### Data Encoding
```Python
le=LabelEncoder()
df['FuelType']=le.fit_transform(df['FuelType'])
```

##### Data Scaling
```Python
scl=MinMaxScaler()
df[['Age']]=scl.fit_transform(df[['Age']])
```

##### Data Visualization:
```Python
data = df.drop(columns=['Automatic'])
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

sns.lineplot(x='FuelType',y='Price',data=df)
plt.title('Average Car Price by FuelType')
plt.xlabel('Fuel Type')
plt.xticks([0,1,2],['Petrol','Diesel','CNG'])
plt.ylabel('Average Price')
plt.grid(True)
plt.show()
```
<table>
<tr>
<td>
  
**Heat Map**
<img src="https://github.com/user-attachments/assets/a7dcb470-6d9e-456e-80e4-222346b518df">  

</td> 
<td valign=top>

**Line Plot** 
<img src="https://github.com/user-attachments/assets/e6801757-4f79-40a1-8c7d-6ac57a8291a8">

</td>
</tr> 
</table>

### Result:
Thus Data analysis and Data preprocessing implemeted using a dataset.
<br>
<br><br><br><br><br><br><br>
**Developed By: ROHIT JAIN D - 212222230120**
