#!/usr/bin/env python
# coding: utf-8

# ## Author: Christian Themin
# ## Starwars Movie Analysis with Python
# 

# This project examines Starwars movie data by performing the first steps of the data science process, including data cleansing and exploration.
# 
# The file contains the data behind America's Favorite Star Wars Movies (and Least Favorite Characters). The authors collected the data by running a poll through SurveyMonkey Audience, surveying 1,186 respondents.
# 
# The analysis will help to define interesting metrics in answering client's questions as follow:
# 1. How many people have seen the movie?
# 2. Which episode of the movie has the highest rank?
# 3. Which Starwars character is the most favorite?
# 4. What's the average income of people who watched the movie?
# 5. Explore people's Demographic towards the Starwars Character

# ## Data Extraction & Discovering

# In[39]:


# Import the required library for analyis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[40]:


# Retrieve the data and rename the columns
filename = 'StarWars.csv'
starwars = pd.read_csv(filename, sep=',', decimal='.', skiprows=2,
                      names=['Resp ID', 'Have you seen the movie?', 'Are you a starwars fan?', 
                            'Seen starwars Ep1', 'Seen starwars Ep2','Seen starwars Ep3',
                            'Seen starwars Ep4', 'Seen starwars Ep5', 'Seen starwars Ep6',
                            'Rank starwars Ep1', 'Rank starwars Ep2', 'Rank starwars Ep3', 
                            'Rank starwars Ep4', 'Rank starwars Ep5', 'Rank starwars Ep6', 
                            'Han', 'Luke', 'Princess', 'Anakin', 'Obi', 'Emperor', 'Darth',
                            'Lando', 'Boba', 'C3P0', 'R2D2', 'Jar', 'Padme', 'Yoda',
                            'Which character shot first?', 'Are you familiar with EU?', 'Are you a fan of EU?',
                            'Are you a fan of startrek?', 'Gender', 'Age', 'Income', 'Education', 'Location'])


# In[41]:


# Check the first few rows of the data
starwars.head()


# In[42]:


# Perform descriptive analysis
starwars.describe()


# In[43]:


# Check the data types
starwars.dtypes


# ## Data Cleaning & Exploration

# ### 1. Explore the variable "Have you seen the movie?".

# In[44]:


# Check the data
starwars['Have you seen the movie?'].value_counts(dropna=False)


# In[45]:


# Remove the extra-white space
starwars['Have you seen the movie?'] = starwars['Have you seen the movie?'].str.strip()


# In[46]:


# Recheck the value
starwars['Have you seen the movie?'].value_counts(dropna=False)


# In[47]:


# Plot the data
starwars['Have you seen the movie?'].value_counts().plot(kind='pie', autopct = '%.2f')
plt.show()


# 78.92% of the participants have seen the movie.<br>
# 21.08% of the participants have not seen the movie.

# ## 2. Explore the variable "Are you a starwars fan?".

# In[48]:


# Check the data
starwars['Are you a starwars fan?'].value_counts(dropna=False)


# In[49]:


# Replace the invalid value to the correct one
starwars['Are you a starwars fan?'].replace('Yess', 'Yes', inplace=True)
starwars['Are you a starwars fan?'].replace('Noo', 'No', inplace=True)
starwars['Are you a starwars fan?'].fillna(value = 'No answer', inplace=True)


# In[50]:


# Recheck the data
starwars['Are you a starwars fan?'].value_counts(dropna=False)


# In[51]:


# Plot the data
starwars['Are you a starwars fan?'].value_counts().plot(kind='pie', autopct='%.02f')
plt.show()


# 46.54% of the participants are Starwars fans.<br>
# 23.95% of the participants are not Starwars fans.<br>
# 29.51% of the participants did not answer.

# ## 3. Explore the variable "Seen starwars Ep1" to "Seen starwars Ep6".

# These variables are boolean type of data, thus, we can set the value to 0 for not seen and 1 for seen.

# In[52]:


# Set 0 and 1 value on the seen starwars Ep 1 to 6
starwars.loc[:,'Seen starwars Ep1':'Seen starwars Ep6'] = np.where(starwars.loc[:, 'Seen starwars Ep1':'Seen starwars Ep6'].isnull(), 0, 1)


# In[53]:


# Group the data into a new variable
seen_movies = starwars.loc[:, 'Seen starwars Ep1':'Seen starwars Ep6']


# In[54]:


seen_movies.head()


# In[55]:


# Summarise the total of the value
seen_movies.sum()


# In[56]:


# Plot the data
seen_movies.sum().sort_values().plot(kind='barh')
plt.show()


# The result of our survey shows that Starwars Episode 5 is the most seen movie compared to the other Starwars episodes.

# ## 4. Explore the variables of Movie ranking fom Ep 1 to Ep 6.

# In[57]:


# Group the variables into one
group_rank = starwars.loc[:,'Rank starwars Ep1':'Rank starwars Ep6']


# In[58]:


# Discover the average value of each variable
group_rank.mean()


# In[59]:


# Plot the value
group_rank.mean().sort_values().plot(kind='barh')
plt.show()


# Starwars Episode 3 has the highest rank compared to the other Starwars episodes.

# ## 5. Explore the variables of all Starwars Characters.

# In[60]:


# Check the data
starwars['Han'].value_counts()


# #### From the result above, we can determine that the value of this variable is an ordinal type of data, we need to reassign the value to numeric for further analysis.

# In[61]:


# Apply new values to the variable according to the rating
mask_character = {'Very favorably': 5,
                  'Somewhat favorably': 4,
                  'Neither favorably nor unfavorably (neutral)': 3,
                  'Somewhat unfavorably': 2,
                  'Very unfavorably': 1,
                  'Unfamiliar': 0
                 }


# In[62]:


# Get the index number by column
starwars.columns.get_loc('Han')


# In[63]:


# Get the index number by column
starwars.columns.get_loc('Yoda')


# In[64]:


# Apply the mask range from 15:29, index no 29 will not be masked, iloc rules
for i in list(range(15,29)):
    starwars.iloc[:, i] = starwars.iloc[:, i].map(mask_character)


# In[65]:


# Recheck the data
starwars['Han'].value_counts(dropna=False)


# In[66]:


starwars['Yoda'].value_counts(dropna=False)


# In[67]:


# Group the data for further analysis
group_char = starwars.loc[:, 'Han':'Yoda']


# In[68]:


# Plot the data
group_char.mean().sort_values().plot(kind='barh')
plt.show()


# Han is the most favorite character amongst all other Starwars characters.

# ## 6. Explore the variable of "Which character shot first?".

# In[69]:


# Check the value of the variable
starwars['Which character shot first?'].value_counts(dropna=False)


# In[70]:


# Plot the data
starwars['Which character shot first?'].value_counts().sort_values().plot(kind='barh')
plt.show()


# ## 7. Explore the variables of:
# "Are you familiar with the Expanded Universe (EU)?" <br>
# "Are you a fan of EU?"<br>
# "Are you a fan of startrek?"<br>

# In[71]:


# Check the value of the variable
starwars['Are you familiar with EU?'].value_counts(dropna=False)


# In[72]:


# Check the value of the variable
starwars['Are you a fan of EU?'].value_counts(dropna=False)


# In[73]:


# Replace the typo value
starwars['Are you a fan of EU?'].replace('Yess', 'Yes', inplace=True)


# In[74]:


# Check the value of the variable
starwars['Are you a fan of startrek?'].value_counts(dropna=False)


# In[75]:


# Replace the typo value
starwars['Are you a fan of startrek?'].replace('no ', 'No', inplace=True)
starwars['Are you a fan of startrek?'].replace('Noo', 'No', inplace=True)
starwars['Are you a fan of startrek?'].replace('yes', 'Yes', inplace=True)


# ### Create dataframes for each value_counts variable

# In[76]:


familiar_EU = starwars['Are you familiar with EU?'].value_counts().to_frame('Familiar with EU').transpose()
familiar_EU


# In[77]:


fan_EU = starwars['Are you a fan of EU?'].value_counts().to_frame('Fan of EU').transpose()
fan_startrek = starwars['Are you a fan of startrek?'].value_counts().to_frame('Fan of Startrek').transpose()


# In[78]:


# To change the axis name, use this code
#fan_startrek = starwars['Are you a fan of startrek?'].value_counts().rename_axis('Fan_of_Startrek').to_frame('counts').transpose()
#fan_startrek


# In[416]:


merged_EU_startrek = pd.concat([familiar_EU, fan_EU, fan_startrek], axis=0)
merged_EU_startrek


# In[417]:


# Plot of merged fans
merged_EU_startrek.plot(kind='barh')
plt.title("Plot fo Merged Fans")
plt.show()


# ## 8. Explore the variables of:
# Gender, Age, Income, and Education

# #### Gender

# In[98]:


# Check the value of gender
starwars['Gender'].value_counts(dropna=False)


# In[99]:


# Replace the typo values to the correct ones
starwars['Gender'].replace('female', 'Female', inplace = True)
starwars['Gender'].replace('male', 'Male', inplace = True)
starwars['Gender'].replace('F', 'Female', inplace = True)


# In[418]:


# Group the gender count into a variable
gender_count = starwars['Gender'].value_counts()
gender_count


# #### Age

# In[173]:


# Check the value of age
starwars['Age'].value_counts(dropna=False)


# In[174]:


# Replace the typo values to the correct ones
starwars['Age'].replace('500', '45-60', inplace=True)


# In[248]:


# Group the gender count into a variable
age_count = starwars['Age'].value_counts()
age_count


# In[419]:


# Plot of Gender and Age
gender_count.plot(kind='barh', color=['pink','lightblue'])
plt.title("Plot of Gender")
plt.show()

age_count.plot(kind='barh', color=('lightblue', 'skyblue'))
plt.title("Plot of Age")
plt.show()


# #### Income

# In[265]:


# Check the value of income
starwars['Income'].value_counts(dropna=False)


# In[266]:


# Group the income count into a variable
income_count = starwars['Income'].value_counts()
income_count


# #### Education

# In[270]:


# Check the value of Education
starwars['Education'].value_counts(dropna=False)


# In[272]:


# Group the education count into a variable
education_count = starwars['Education'].value_counts()
education_count


# In[399]:


# Plot of Income and Education
income_count.plot(kind='barh', color=('gray', 'lightgray'))
plt.title("Plot of Income")
plt.show()

education_count.plot(kind='barh', color=['burlywood'])
plt.title("Plot of Education")
plt.show()


# ## 9. Explore the variable of Location

# In[420]:


# Check the value of the data
starwars['Location'].value_counts()


# In[421]:


# Plot the location
starwars['Location'].value_counts().plot(kind='barh', color=['violet', 'pink'])
plt.show()


# ## Discover relationship between variables

# ### Explore the relationship between being a Starwars fan and Gender

# In[422]:


# Map the Starwars fan and not fan as 0 and 1
mask_fan_or_not = {'Yes': 1, 'No':0}
starwars['Are you a starwars fan?'] = starwars['Are you a starwars fan?'].map(mask_fan_or_not)


# In[423]:


starwars['Are you a starwars fan?'].value_counts()


# In[425]:


# Mask for gender
mask_female = starwars['Gender'] == 'Female'
mask_male = starwars['Gender'] == 'Male'


# In[426]:


# Value counts for each gender
gender_count = starwars['Gender'].value_counts()


# In[427]:


# Select the fan that is male
male_fan = starwars.loc[mask_male, 'Are you a starwars fan?'].value_counts()
# Select the fan that is female
female_fan = starwars.loc[mask_female, 'Are you a starwars fan?'].value_counts()


# In[428]:


# For every male fan of gender count as Male, For every female fan of gender count as female
rate = [male_fan[1]/float(gender_count['Male']), female_fan[1]/float(gender_count['Female'])]


# In[429]:


plt.bar(list(range(2)), rate, color='lightblue')
plt.xticks(list(range(2)), ['Male', 'Female'])
plt.xlabel('Gender')
plt.ylabel('Frequency of Fan number')
plt.title('Gender comparison of Starwars Fan')
plt.show()


# In[430]:


# Percentage of Male Starwars Fan
male_fan[1]/float(gender_count['Male'])*100


# In[431]:


# Percentage of Female Starwars Fan
female_fan[1]/float(gender_count['Female'])*100


# ## Explore the relationship between Seen the movie with Starwars Fan

# In[432]:


starwars.columns


# In[433]:


starwars['Have you seen the movie?'] = starwars['Have you seen the movie?'].map(mask_fan_or_not)


# In[434]:


starwars['Have you seen the movie?'].value_counts()


# In[435]:


# fan count
SW_fan_count = starwars['Are you a starwars fan?'].value_counts()
# Yes, I am a fan
I_am_a_fan = starwars['Are you a starwars fan?'] == 1
I_am_not_a_fan = starwars['Are you a starwars fan?'] == 0


# In[436]:


I_am_a_fan.value_counts()


# In[437]:


# Drop na is important here, or the plot will be flat later
seen_fan = starwars.dropna().loc[I_am_a_fan, 'Have you seen the movie?'].value_counts()
seen_not_fan = starwars.dropna().loc[I_am_not_a_fan, 'Have you seen the movie?'].value_counts()


# In[438]:


rate_2 = [seen_fan[1]/float(SW_fan_count[1]), seen_not_fan[1]/float(SW_fan_count[0])]


# In[439]:


plt.bar(list(range(2)), rate_2, color='grey')
plt.xticks(list(range(2)), ['Yes', 'No'])
plt.xlabel('Fan')
plt.ylabel('Seen the movies?')
plt.title('Seen the Movie and Fan of Starwars')


# In[440]:


# Percentage of seen movie and a Fan
seen_fan[1]/float(SW_fan_count[1])*100


# In[441]:


# Percentage of seen movie but not a fan
seen_not_fan[1]/float(SW_fan_count[1])*100


# ## Explore the relationship between Familiar with Expanded Universe and A fan of EU

# In[442]:


# Mapping the familiar or not into Boolean value
starwars['Are you familiar with EU?'] = starwars['Are you familiar with EU?'].map(mask_fan_or_not)


# In[443]:


starwars['Are you familiar with EU?'].value_counts()


# In[444]:


# Select all fan and count the total
fan_count = starwars['Are you a fan of EU?'].value_counts()
# Group the fan
mask_fan = starwars['Are you a fan of EU?'] == 'Yes'
# Group the not fan
mask_not_fan = starwars['Are you a fan of EU?'] == 'No'


# In[445]:


# Select the fan that is familiar with EU
fan_and_familiar = starwars.dropna().loc[mask_fan, 'Are you familiar with EU?'].value_counts()
# Select the not fan but is familiar with EU
not_fan_but_familiar = starwars.dropna().loc[mask_not_fan, 'Are you familiar with EU?'].value_counts()


# In[446]:


rate_1 = [fan_and_familiar[1]/float(fan_count['Yes']), [not_fan_but_familiar[1]/float(fan_count['No'])]]


# In[447]:


plt.bar(list(range(2)), rate_1, color='pink')
plt.xticks(list(range(2)), ['Yes', 'No'])
plt.xlabel('Fan of EU')
plt.ylabel('Familiar with EU')
plt.title('Familiar with EU and Fan of EU')


# In[448]:


# Percentage of fan and familiar
fan_and_familiar[1]/float(fan_count['Yes'])*100


# In[449]:


# Percentage of not fan but familiar
not_fan_but_familiar[1]/float(fan_count['No'])*100


# ## Explore people's Demographic towards Starwars Character

# In[450]:


group_char.hist(figsize=(15,15))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




