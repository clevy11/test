import pandas as pd


# #View Dirty Data Set
# df = pd.read_csv('Health.csv')
# print(df.to_string())
# input("Press Enter to Delete Empty Cells")
#Delete Empty Cells
# new_df = df.dropna()
# print(new_df.to_string())
# input("Press Enter to remove empty rows")
# #Delete Empty Rows
# df.dropna(inplace = True)
# print(df.to_string())
# input("Press Enter to replace Null Values with a Number")
# #Replace Null Values with 130
# df.fillna(130, inplace = True)
# print(df.to_string())
# #Replace Null Values h 130 on Specified Column
# df = pd.read_csv('health.csv')
# df["Calories"].fillna(130, inplace = True)
# print(df.to_string())
# input("Press Enter if you want to replace empty value with Mean…")
# #Replace with Mean
# df = pd.read_csv('health.csv')
# x = df["Calories"].mean()
# df["Calories"].fillna(x, inplace = True)
# print(df.to_string())
# # input("Press Enter if you want to replace empty value with Median…")
# #Replace with Median
# df = pd.read_csv('health.csv')
# x = df["Calories"].median()
# df["Calories"].fillna(x, inplace = True)
# print(df.to_string())
# input("Press Enter if you want to replace empty value with Mode…")
# # Replace with Mode
# df = pd.read_csv('health.csv')
# x = df["Calories"].mode()[0]
# df["Calories"].fillna(x, inplace = True)
# print(df.to_string())

# df = pd.read_csv('health.csv')
# df['Date'] = pd.to_datetime(df['Date'])
# print(df.to_string())
# df = pd.read_csv('health.csv')
# df.loc[7,'Duration'] = 45
# print(df.to_string())
# input("Press Enter to Replace Value by Loop through the Column…")
# #Replace Value by Loop through the Column
# df = pd.read_csv('health.csv')
# for x in df.index:
# if df.loc[x, "Duration"] > 120:
# df.loc[x, "Duration"] = 120
# print(df.to_string())

# df = pd.read_csv('data.csv')
# print(df.duplicated())
# #Removing Duplicates
# df = pd.read_csv('data.csv')
# df.drop_duplicates(inplace = True)
# print(df.to_string())


# # Read the CSV file
# df = pd.read_csv('health.csv')
#
# # Replace Null Values with 130
# df["Calories"] = df["Calories"].fillna(130)
#
# # Replace with Mean
# x_mean = df["Calories"].mean()
# df["Calories"].fillna(x_mean, inplace=True)
#
# # Replace with Median
# x_median = df["Calories"].median()
# df["Calories"].fillna(x_median, inplace=True)
#
# # Replace with Mode
# x_mode = df["Calories"].mode()[0]
# df["Calories"].fillna(x_mode, inplace=True)
#
# # Save the DataFrame to the same CSV file
# df.to_csv('updated_health.csv', index=False)



#
# # Define your data as a two-dimensional list
# data = [
#     ["Alice", 25, "Engineer"],
#     ["Bob", 30, "Manager"],
#     ["Charlie", 35, "Data Scientist"]
# ]
#
# # Define column names
# columns = ["Name", "Age", "Occupation"]
#
# # Create the DataFrame
# df = pd.DataFrame(data, columns=columns)
#
# # Print the DataFrame
# print(df)
# df.to_csv('clb.csv', index=False)


# ?Load Files Into a DataFrame
# Replace 'My_Data.xlsx' with the correct file path to your Excel file
# df = pd.read_excel('My_Data.xlsx')

# Print the DataFrame
# print(df.to_string())

# Describe the Data Frame
# Print descriptive statistics of the DataFrame
# print(df.describe())
# #
# # Sort the Data Frame
# # Sort the DataFrame by the 'SNAMES' column
# print(df.sort_values('SNAMES '))
#
# input("Enter To Filter the Dataframe...")
# # df = pd.read_excel('My_Data.xlsx')
# print (df.filter(["SNAMES ","QUIZZES "]))
# #Filter Columns Contain aA…
# input ("Press Enter to Filter Columns Contain aA..")
# print (df.filter(regex ='[aA]'))
# #Group Data Frame
# input ("Press Enter to Group Student by Semester..")
# print (df.groupby(['Semester']).mean())
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Import Data from Excel file
Grade_data=pd.read_excel('Data.xlsx')
#Learning with Decision Classifier
X=Grade_data.drop (columns=['SNAMES ','Total Marks','Marks /20', 'Grading '])
y=Grade_data['Grading ']

X_train, x_test, y_train, y_test=train_test_split (X,y, test_size=0.2)
Decision_tree_model= DecisionTreeClassifier()
Logistic_regression_Model=LogisticRegression(solver='lbfgs',max_iter=10000)
SVM_model=svm.SVC(kernel='linear')
RF_model=RandomForestClassifier(n_estimators=100)
Decision_tree_model.fit(X_train, y_train)
Logistic_regression_Model.fit(X_train, y_train)
SVM_model.fit(X_train, y_train)
RF_model.fit(X_train, y_train)
DT_Prediction =Decision_tree_model.predict(x_test)
LR_Prediction =Logistic_regression_Model.predict(x_test)
SVM_Prediction =SVM_model.predict(x_test)
RF_Prediction =RF_model.predict(x_test)
DT_score=accuracy_score(y_test, DT_Prediction)
lR_score=accuracy_score(y_test, LR_Prediction)
SVM_score=accuracy_score(y_test, SVM_Prediction)
RF_score=accuracy_score(y_test, RF_Prediction)
print ("Decistion Tree accuracy =", DT_score*100,"%")
print ("Logistic Regression accuracy =", lR_score*100,"%")
print ("Suport Vector Machine accuracy =", SVM_score*100,"%")
print ("Random Forest accuracy =", RF_score*100,"%")
from sklearn import svm
#
#
# import joblib
# Grade_data=pd.read_excel('Data.xlsx')
# # Learning with the Model
# X=Grade_data.drop (columns=['SNAMES ','Total Marks','Marks /20', 'Grading '])
# y=Grade_data['Grading ']
# model= svm.SVC(kernel='linear')
# model.fit(X.values, y)
# joblib.dump(model, 'grade-recommender.joblib')

# Quiz=int (input ("Enter Quiz Marks :"))
# Assgn= input ("Enter Assignment Marks: ")
# Mid=int (input ("Enter Mid Exam Marks Marks :"))
# Final= input ("Enter Final Exam Marks: ")
# #Predict from the created model
# model=joblib.load('grade-recommender.joblib')
# predictions = model.predict ([[Quiz,Assgn,Mid,Final]])
# print("The Grade you will obtain is:",predictions )
