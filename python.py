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
    baseDT = DecisionTreeClassifier()
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
    plot_tree(topDT, filled=True, feature_names=X_penguins.columns, class_names=y_penguins.unique()) # topDT is the tree model that we are showing, Filled is for color, 
    plt.show()
    #plot_distribution(penguins, "species", "penguins-classes-topDT.png")
    
    #4c) Base MLP for Penguins
    baseMLP = MLPClassifier(hidden_layer_sizes=(100,100), activation='logistic', solver='sgd')
    baseMLP.fit(X_train_penguins, y_train_penguins) #trains the data based on both training subsets of input and output
    y_pred_baseMLP = baseMLP.predict(X_test_penguins) #provides an output (species) prediction based on a input test subset
    print("Base-MLP Penguin Performance\n", classification_report(y_test_penguins, y_pred_baseMLP, zero_division=1)) #evaluating the predicted species subset versus the actual species subset
    
    #4d)
    #Instructions on how to conduct search (Mapping)
    MLPparams = {
    'activation': ['logistic', 'tanh', 'relu'], #activation functions
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)], #Network architectures
    'solver': ['adam', 'sgd'] #Solver for weight distribution
    }
    #Use GridSearchCV to perform an exhaustive search using acuracy as the scoring.
    topMLP = GridSearchCV(MLPClassifier(), MLPparams, scoring='accuracy')
    #To train the model
    topMLP.fit(X_train_penguins, y_train_penguins)
    print('Top-MLP Best Parameters:', topMLP.best_params_)
    y_pred_topMLP = topMLP.predict(X_test_penguins)
    print('Top-MLP Performance for Penguins:')
    print(classification_report(y_test_penguins, y_pred_topMLP, zero_division=1))



def abaloneSteps(abalone):
    #2:
    plot_distribution(abalone, "Type", "abalone-classes.png")
    
    #3 split data
    x = abalone.drop('Type', axis=1)
    y = abalone['Type']
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    #4a): Base-DT for abalone
    baseDT = DecisionTreeClassifier()
    baseDT.fit(x_train, y_train)
    y_pred_baseDT = baseDT.predict(x_test)
    print('Base-DT Performance for abalone:')
    #Comparing true results with prediction
    print(classification_report(y_test, y_pred_baseDT, zero_division=1))


    #4b): Top-DT for Abalone
    # Define hyperparameters to search for
    param_grid = {
        'criterion': ['gini', 'entropy'], #Using gini or entorpy as mention in the Insturctions for the decisionTree
        'max_depth': [None, 2, 10],  # 2 different values of our choose including a ”None” option. Values were chosen random.
        'min_samples_split': [2, 3, 4],  # 3 different values of our choice. Values were chosen random.
    }

    # Create a Decision Tree classifier like in 4a from Scikit-Learn
    topDT = DecisionTreeClassifier()

    # Finding the hyperparameters using a GridSearchCv function from Scikit-Learns.
    grid_search = GridSearchCV(topDT, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    # Getting the best hyperparameters based on our grid_search.
    best_params = grid_search.best_params_ #the .best_params_ gives the best results on the hold out data
    print('Best Hyperparameters:', best_params)

    # Train the Top-DT with the best hyperparameters
    topDT = DecisionTreeClassifier(**best_params) #The two stars **unpacks the variable, equivalent to placing it as DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=5). Though, we don't know the best parameters. 
    topDT.fit(x_train, y_train)
    y_pred_topDT = topDT.predict(x_test)

    print('Top-DT Performance for Penguins:')
    print(classification_report(y_test, y_pred_topDT, zero_division=1))

    #Showing the decision tree graphically (depth is restricted for visualization purposes)
    plt.figure(figsize=(10, 8)) #10 inches by 8 inches. Reduce the figure size if the monitor/screen is small.
    plot_tree(topDT, filled=True, feature_names=x.columns, class_names=y.unique()) # topDT is the tree model that we are showing, Filled is for color, 
    plt.show()
    
    #4c) Base MLP for abalone
    baseMLP = MLPClassifier(hidden_layer_sizes=(100,100), activation='logistic', solver='sgd')
    baseMLP.fit(x_train, y_train) #trains the data based on both training subsets of input and output
    y_pred_baseMLP = baseMLP.predict(x_test) #provides an output (type) prediction based on a input test subset
    print("Base-MLP abalone Performance\n", classification_report(y_test, y_pred_baseMLP, zero_division=1)) #evaluating the predicted type subset versus the actual type subset
    
    #4d)
    #Instructions on how to conduct search (Mapping)
    MLPparams = {
    'activation': ['logistic', 'tanh', 'relu'], #activation functions
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)], #Network architectures
    'solver': ['adam', 'sgd'] #Solver for weight distribution
    }
    #Use GridSearchCV to perform an exhaustive search using acuracy as the scoring.
    topMLP = GridSearchCV(MLPClassifier(), MLPparams, scoring='accuracy')
    #To train the model
    topMLP.fit(x_train, y_train)
    print('Top-MLP Best Parameters:', topMLP.best_params_)
    y_pred_topMLP = topMLP.predict(x_test)
    print('Top-MLP Performance for Penguins:')
    print(classification_report(y_test, y_pred_topMLP, zero_division=1))

   
def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))  # Get the directory/filepath of the script
    abalone_file_path = os.path.join(script_directory, "abalone.csv")
    penguins_file_path = os.path.join(script_directory, "penguins.csv")
    penguin_data = pd.read_csv(penguins_file_path)
    abalone_data = pd.read_csv(abalone_file_path)
    
    while(True):
        user_input = input("Please select one of the options below to check a file.\n(1) abalone.csv\n(2) penguins.csv\n")
        user_choice = string_to_int(user_input)
        if user_input and user_choice <= 3:
            if user_choice == 1:
                print("abalone.csv has been selected")
                abaloneSteps(abalone_data)
                #read_csv(abalone_data) -- old
                break
            elif user_choice == 2:
                print("penguins.csv has been selected")
                penguinsSteps(penguin_data)
                #read_csv(penguins_file_path)  -- old
                break  
            else:
                print("[Error]: Invalide option has been selected.")
        else:
            print("[Error]: User input is empty. Please enter a valid file.")

if __name__ == "__main__":
    main()