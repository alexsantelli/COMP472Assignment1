#Will's
import os
import sys
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html    
from sklearn.model_selection import train_test_split #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
from sklearn.tree import DecisionTreeClassifier #https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.neural_network import MLPClassifier #https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
from sklearn.model_selection import GridSearchCV #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.tree import plot_tree #https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html#sklearn.tree.plot_tree   This will be used in 4.B


def string_to_int(str) -> int:
    try:
        output = int(str)
    except ValueError:
        print("invalid Input. Enter Integer value\n")
        return 5
    return output

def plot_instance_class_distribution(data, dataset_name):
    if dataset_name == 'penguins Dataset':
        class_counts = data['species'].value_counts() if dataset_name == 'penguins' else data['sex'].value_counts()
        class_percentage = class_counts / len(data)
        plt.bar(class_counts.index, class_percentage)
        plt.xlabel("Class")
        plt.ylabel("Percentage")
        plt.title(f"{dataset_name} Class Distribution")
        #plt.savefig(f"{dataset_name}-classes.gif")
        plt.show()

# Plot the distribution of the 'species' column

def plot_distribution(type: pd, class_column: str, file_name: str):
    # Calculate the percentage of instances in each class
    class_distribution = type[class_column].value_counts(normalize=True) * 100
    
    # Create a bar plot of the class distribution
    ax = class_distribution.plot(kind='pie', autopct = '%.2f%%')
    plt.title(f'{class_column} Class Distribution')
    
    # Save the plot as a PNG file
    #plt.savefig(file_name)
    plt.show()

def read_csv(file_path):
    try:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                print(row)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def abaloneSteps(abalone):
    #2:
    plot_distribution(abalone, "Type", "abalone-classes.png")

def penguinsSteps(penguin_data):
    #1: Convert Categorical Features for Penguins into 1 hot vector
    penguins = pd.get_dummies(penguin_data, columns=['island', 'sex'], drop_first=True) #drop first used to drop the species column
    
    #2:
    plot_distribution(penguins, "species", "penguins-classes.png")
    
    #3: split data
    X_penguins = penguins.drop('species', axis=1)
    y_penguins = penguins['species']
    X_train_penguins, X_test_penguins, y_train_penguins, y_test_penguins = train_test_split(X_penguins, y_penguins)
    #4a): Base-DT for Penguins
    baseDT = DecisionTreeClassifier() #TODO: May need to add random state
    baseDT.fit(X_train_penguins, y_train_penguins)
    y_pred_baseDT = baseDT.predict(X_test_penguins)
    print('Base-DT Performance for Penguins:')
    #Comparing true results with prediction
    print(classification_report(y_test_penguins, y_pred_baseDT, zero_division=1))

    #4b): Top-DT for Penguins
    # Define hyperparameters to search for
    param_grid = {
        'criterion': ['gini', 'entropy'], #Using gini or entorpy as mention in the Insturctions for the decisionTree
        'max_depth': [None, 10, 100],  # 2 different values of our choose including a ”None” option. Values were chosen random.
        'min_samples_split': [2, 5, 20],  # 3 different values of our choice. Values were chosen random.
    }

    # Create a Decision Tree classifier like in 4a from Scikit-Learn
    topDT = DecisionTreeClassifier()

    # Finding the hyperparameters using a GridSearchCv function from Scikit-Learns.
    grid_search = GridSearchCV(topDT, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_penguins, y_train_penguins)

    # Getting the best hyperparameters based on our grid_search.
    best_params = grid_search.best_params_ #the .best_params_ gives the best results on the hold out data
    print('Best Hyperparameters:', best_params)

    # Train the Top-DT with the best hyperparameters
    topDT = DecisionTreeClassifier(**best_params) #The two stars **unpacks the variable, equivalent to placing it as DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=5). Though, we don't know the best parameters. 
    topDT.fit(X_train_penguins, y_train_penguins)
    y_pred_topDT = topDT.predict(X_test_penguins)

    print('Top-DT Performance for Penguins:')
    print(classification_report(y_test_penguins, y_pred_topDT, zero_division=1))

    #Showing the decision tree graphically (depth is restricted for visualization purposes)
    plt.figure(figsize=(10, 8)) #10 inches by 8 inches. Reduce the figure size if the monitor/screen is small.
    plot_tree(topDT, filled=True, feature_names=X_penguins.columns, class_names=y_penguins.unique())
    plt.show()
    #plot_distribution(penguins, "species", "penguins-classes-topDT.png")

def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))  # Get the directory/filepath of the script
    abalone_file_path = os.path.join(script_directory, "abalone.csv")
    penguins_file_path = os.path.join(script_directory, "penguins.csv")
    penguin_data = pd.read_csv(penguins_file_path)
    abalone_data = pd.read_csv(abalone_file_path)
    

    
    while(True):
        user_input = input("Please select one of the options below to check a file.\n(1) abalone.csv\n(2) penguins.csv\n(3) custome file (File path is required)\n")
        user_choice = string_to_int(user_input)
        if user_input and user_choice <= 3:
            if user_choice == 1:
                print("abalone.csv has been selected")
                plot_instance_class_distribution(abalone_data, 'Abalone Dataset')
                abaloneSteps(abalone_data)
                #read_csv(abalone_data) -- old
                break
            elif user_choice == 2:
                print("penguins.csv has been selected")
                plot_instance_class_distribution(penguin_data, 'Penguins Dataset')
                penguinsSteps(penguin_data)
                #read_csv(penguins_file_path)  -- old
                break
            elif user_choice == 3:
                while(True):
                    user_input_custom = input("Custom file has been selects\nPlease enter the file path and name that you want to review")
                    if user_input_custom: 
                        #read_csv(user_input)  -- old
                        break
                    else:
                        print("[Error]: User input is empty. Please enter a valid file.")    
            else:
                print("[Error]: Invalide option has been selected.")
        else:
            print("[Error]: User input is empty. Please enter a valid file.")

if __name__ == "__main__":
    main()