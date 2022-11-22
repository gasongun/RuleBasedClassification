
############### Import Libraries ###############

import pandas as pd
import os
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 20)

############### Import The Data ###############

def load_dataset(data):
    path = os.getcwd()
    return pd.read_csv(path+data+".csv")

df = load_dataset("persona")
df.head()

############### TASK 1 ###############

# Question 1
# Summarize the data and explain the general information

def summary_data(df):
    print("  Head   :")
    print(df.head())
    print("  Shape   :")
    print(df.shape)
    print("  Describe   :")
    print(df.quantile([0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)
    print("  Null   :")
    print(df.isnull().sum())
    print("  Columns   :")
    print(df.columns)
    print("  Duplicate   :")
    print(df.duplicated().sum())
    print("  Index max   :")
    print(df.index.max())
    print("  Info   :")
    print(df.info())

summary_data(df)

"""
Describe function shows us scale variable features. If there is any outlier and extreme values, we can detect with this function. 
Dataset looks stabil and also no null case in it. 
Info function gives type of columns for check variables are of the appropriate type.
"""

# Function for some of next questions

def expCatVar(df,col,nunique_count=False,value_count=False,):
    if nunique_count:
        print("Unique count: ", df[col].nunique(), "\n")
    if value_count:
        print(pd.DataFrame({"Frequency": df[col].value_counts(),
                            "Ratio": (df[col].value_counts() / len(df)) * 100}))

# Question 2
# How many unique Source in data? What is the frequencies?

expCatVar(df, 'SOURCE', nunique_count = True, value_count = True)

# Question 3
# How many unique PRICE in data?

expCatVar(df, 'PRICE', nunique_count = True)

# Question 4
# What is the unique categories and frequencies every categorical variables?

for i in df.loc[:,df.dtypes == 'O']:
    print(i, 'has' ,df[i].nunique(), 'unique group and \n', df[i].value_counts(), 'value')

# *** Second Way ***
for i in df.loc[:,df.dtypes == 'O']:
    print(i, 'has' ,expCatVar(df, i, nunique_count = True), 'unique group and \n', expCatVar(df, i, value_count = True))


# Question 5
# How many sales made from PRICE categories?

expCatVar(df, 'PRICE', value_count = True)

# Question 6
# How many sales made from COUNTRY categories?

expCatVar(df, 'COUNTRY', value_count = True)


# Function for some of the next questions
def groupFunction(df, col, agg_col, count = True, mean = False, min = False, max = False, sum = False):

    """
    This function to use for aggregating the data. Function gives count aggregation as default.
    Dataframe, column and variable for aggregation are required for the function to work.

    Parameters
    ----------
    df: dataframe
    col: Use to determine the groups for the groupby
    agg_col: Determine the variables to use for aggregating
    count: Gives the count of the agg_col
    mean: Gives the mean of the agg_col
    min: Gives the minimum value of the agg_col
    max: Gives the maximum value of the agg_col
    sum: Gives the sum of the agg_col

    """
    if count:
        print("Count : ")
        return df.groupby(col).agg({agg_col: "count"})
    if mean:
        print("Mean : ")
        return df.groupby(col).agg({agg_col: "mean"})
    if sum:
        print("Sum : ")
        return df.groupby(col).agg({agg_col: "sum"})
    if min:
        print("Min : ")
        return df.groupby(col).agg({agg_col: "min"})
    if max:
        print("Max : ")
        return df.groupby(col).agg({agg_col: "max"})



# Question 7
# What is total PRICE of the COUNTRY categories?

groupFunction(df, "COUNTRY", "PRICE", count = False, sum = True)

# *** Second Way Without Function ***
df[["COUNTRY","PRICE"]].groupby("COUNTRY").agg({"sum"})


# Question 8
# What is count of the SOURCE categories?

groupFunction(df, "SOURCE", "SOURCE")

# Question 9
# What is mean PRICE of the COUNTRY categories?

groupFunction(df, "COUNTRY", "PRICE", count = False, mean = True)

# *** Second Way Without Function ***
df[["COUNTRY","PRICE"]].groupby("COUNTRY").mean()

# Question 10
# What is mean PRICE of the SOURCE categories?

groupFunction(df, "SOURCE", "PRICE", count = False, mean = True)

# *** Second Way Without Function ***
df[["SOURCE","PRICE"]].groupby("SOURCE").mean()

# Question 11
# What is mean PRICE of the SOURCE and COUNTRY categories?

groupFunction(df, ["COUNTRY","SOURCE"], "PRICE", count = False, mean = True)

# *** Second Way Without Function ***
df[["SOURCE","COUNTRY","PRICE"]].groupby(["COUNTRY","SOURCE"]).mean()


###############################################

############### TASK 2 ###############

# What is the mean PRICE of the COUNTRY, SOURCE, SEX, AGE?
groupFunction(df, ["COUNTRY","SOURCE", "SEX", "AGE"], "PRICE", count = False, mean = True)

# *** Second Way Without Function ***
df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).mean()


###############################################

############### TASK 3 ###############

# Use a sort method to the output in the previous question and assign to new dataframe

agg_df = groupFunction(df, ["COUNTRY","SOURCE", "SEX", "AGE"], "PRICE", count = False, mean = True).sort_values("PRICE",ascending=False)
agg_df.head(10)


###############################################

############### TASK 4 ###############

# Change the index names to variable names without PRICE

agg_df = agg_df.reset_index()


###############################################

############### TASK 5 ###############

# Change the numerical AGE variable to categorical AGE variable and add the agg_df

agg_df['AGE_GROUP'] = pd.cut(agg_df["AGE"],bins=[0,18,23,30,40,agg_df["AGE"].max()],labels=['0_18','19_23','24_30','31_40','41+'])
agg_df[['AGE_GROUP','AGE']].groupby('AGE_GROUP').agg({"min", "max"})  # to control limits of the groups


###############################################

############### TASK 6 ###############

# Step 1
# Define a level based persona with COUNTRY, SOURCE, SEX and AGE_GROUP variables

agg_df['AGE_GROUP'] = agg_df['AGE_GROUP'].astype(object)
agg_df['customers_level_based'] = agg_df[[col for col in agg_df.columns if agg_df[col].dtypes =='O']].apply(lambda x:'_'.join(x), axis=1).str.upper()


# Step 2
# What is the mean PRICE of singularized persona variables that created in previous question

persona = groupFunction(agg_df, "customers_level_based", "PRICE", count = False, mean = True)

###############################################

############### TASK 7 ###############

# Question 1
# Segment each persona by price level

persona = persona.reset_index()
persona['SEGMENT'] = pd.qcut(persona["PRICE"],4 , labels = ["D","C","B","A"])


# Question 2
# Describe the segment groups
persona[["SEGMENT","PRICE"]].groupby("SEGMENT").agg({'mean','min','max','std','sum','count'})

""" Group A is a best group in these data. They bought the most expensive and more games than the other groups. Group D is opposite, they bought cheap and less games. """

###############################################

############### TASK 8 ###############

# Segment the new customers.

def new_customer():
    COUNTRY = input("What is the first 3 letter in your country?")
    SOURCE = input("Which system uses your phone? Android/IOS")
    SEX = input("What is your gender? Female/Male")
    AGE = int(input("Which group is suitable for your age? '0_18','19_23','24_30','31_40','41+'"))
    ageList = ['0_18','19_23','24_30','31_40','41+']
    for i in range(len(ageList)):
        if AGE == i:
            AGE_ = ageList[i]
    customer_ = COUNTRY.upper() + '_' + SOURCE.upper()  + '_' + SEX.upper() + '_' + AGE_

    print(customer_)
    print("Segment:" + persona[persona["customers_level_based"] == customer_].loc[:,"SEGMENT"].values[0])
    print("Price:" + str(persona[persona["customers_level_based"] == customer_].loc[:, "PRICE"].values[0]))


# Question 1
# What is segment of a 33 years old, android user turkish female? and price?

new_customer()

# inputs are: COUNTRY -> TUR   SOURCE -> ANDROID   SEX -> FEMALE   AGE -> 3
# Segment A, price: 41.8

# Question 2
# What is segment of a 28 years old, ios user french male? and price?

new_customer()

# inputs are: COUNTRY -> FRA   SOURCE -> IOS   SEX -> FEMALE   AGE -> 2
# Segment D, price: 31.5

