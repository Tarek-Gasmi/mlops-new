import pandas as pd
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Set random seed
seed = 42

################################
########## DATA PREP ###########
################################

# Load in the data
df = pd.read_csv("wine_quality.csv")
# Create Classification version of target variable
df['goodquality'] = [1 if x >= 6 else 0 for x in df['quality']]
# Separate feature variables and target variable
X = df.drop(['quality','goodquality'], axis = 1)
y = df['goodquality']
# Split into train and test sections
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


#################################
########## MODELLING ############
#################################

# Fit a model on the train section
model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=seed, oob_score=True )
model.fit(X_train, y_train)
predictions = model.predict(X_test)
# Report training set score
train_score = model.score(X_train, y_train) * 100
# Report test set score
test_score = model.score(X_test, y_test) * 100
# Report RMSE
mse = mean_squared_error(y_test, predictions)
# Report RMSE
rmse = np.sqrt(mse)
# Report Classification error report
error_report = classification_report(y_test, predictions)



# Write scores to a file
with open("metrics_1.txt", 'w') as outfile:
         outfile.write("Training variance explained: %2.1f%%\n" % train_score)
         outfile.write("Test variance explained: %2.1f%%\n" % test_score)
         outfile.write("Root Mean Squared Error: %2.1f%%\n" % rmse)
         outfile.write("Out-of-bag Score: %2.1f%%\n" % model.oob_score_)

with open("metrics_2.txt", 'w') as outfile:
         outfile.write(error_report)

# save FPDF() class into 
# a variable pdf
pdf = FPDF()   
   
# Add a page
pdf.add_page()
   
# set style and size of font 
# that you want in the pdf
pdf.set_font("Arial", size = 15)
  
# open the text file in read mode
f = open("metrics_2.txt", "r")
  
# insert the texts in pdf
for x in f:
    pdf.cell(200, 10, txt = x, ln = 1, align = 'C')
   
# save the pdf with name .pdf
pdf.output("metrics_2.pdf")   

##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################
# Calculate feature importance in random forest
importances = model.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"])
feature_df = feature_df.sort_values(by='importance', ascending=False,)

# image formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Importance',fontsize = axis_fs) 
ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png",dpi=120) 
plt.close()
