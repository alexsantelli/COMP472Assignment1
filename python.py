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

RANGE = 5
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
    plt.savefig(file_name)

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

def evaluate(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, macro_f1, weighted_f1


def report_performance(dataset_name, model_name, accuracy, macro_f1, weighted_f1):
    with open(f'{dataset_name}-performance.txt', 'a') as file:
        file.write(f"{model_name} (D) Accuracy, Macro-Average F1, Weighted-Average F1\n")
        file.write(f"{model_name} Accuracy: {accuracy}\n")
        file.write(f"{model_name} Macro F1: {macro_f1}\n")
        file.write(f"{model_name} Weighted F1: {weighted_f1}\n")
        
def confusionMatrix(dataset_name, y_test, y_pred):
    with open(f'{dataset_name}-performance.txt', 'a') as file:
        file.write("Confusion Matrix:\n")
        file.write(f'{confusion_matrix(y_test, y_pred)}\n')

def classificationReport(dataset_name, y_test, y_pred):
    with open(f'{dataset_name}-performance.txt', 'a') as file:
        file.write("Classification Report:\n")
        file.write(f'{classification_report(y_test, y_pred, zero_division=1)}\n')

def appendperformance(dataset_name, model_name: str, metrics: dict[str, list]):
    avg = np.mean(metrics[model_name])
    avg_variance = np.var(metrics[model_name])
    with open(f'{dataset_name}-performance.txt', 'a') as file:
        file.write(f'{model_name} Average Accuracy: {avg} Variance: {avg_variance}\n')
    

def baseDT(name:str, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    baseDT = DecisionTreeClassifier()
    baseDT.fit(x_train, y_train)
    y_pred_baseDT = baseDT.predict(x_test)
    accuracy, macro_f1, weighted_f1 = evaluate(y_test, y_pred_baseDT)
    #5B)
    confusionMatrix(name,y_test, y_pred_baseDT)
    #5C)
    classificationReport(name,y_test, y_pred_baseDT)
    #5D)
    report_performance(name ,"Base-DT", accuracy, macro_f1, weighted_f1)
    print('Base-DT Performance for Penguins:')
    #Comparing true results with prediction
    print(classification_report(y_test, y_pred_baseDT, zero_division=1))
    plt.figure(figsize=(10,8))
    plot_tree(baseDT, filled=True)
    plt.show()
    return accuracy, macro_f1, weighted_f1

def topDT(name:str, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y)
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
    grid_search.fit(x_train, y_train)

    # Getting the best hyperparameters based on our grid_search.
    best_params = grid_search.best_params_ #the .best_params_ gives the best results on the hold out data
    print('Best Hyperparameters:', best_params)

    # Train the Top-DT with the best hyperparameters
    topDT = DecisionTreeClassifier(**best_params) #The two stars **unpacks the variable, equivalent to placing it as DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=5). Though, we don't know the best parameters. 
    topDT.fit(x_train, y_train)
    y_pred_topDT = topDT.predict(x_test)
    
    accuracy, macro_f1, weighted_f1 = evaluate(y_test, y_pred_topDT)
    #5B)
    confusionMatrix(name,y_test, y_pred_topDT)
    #5C)
    classificationReport(name,y_test, y_pred_topDT)
    #5D)
    report_performance(name ,"Top-DT", accuracy, macro_f1, weighted_f1)
    print('Top-DT Performance for Penguins:')
    print(classification_report(y_test, y_pred_topDT, zero_division=1))

    #Showing the decision tree graphically (depth is restricted for visualization purposes)
    plt.figure(figsize=(10, 8)) #10 inches by 8 inches. Reduce the figure size if the monitor/screen is small.
    plot_tree(topDT, filled=True, feature_names=x.columns, class_names=y.unique().astype(str)) # topDT is the tree model that we are showing, Filled is for color, 
    plt.show()
    return accuracy, macro_f1, weighted_f1

def baseMLP(name:str, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    baseMLP = MLPClassifier(hidden_layer_sizes=(100,100), activation='logistic', solver='sgd')
    baseMLP.fit(x_train, y_train) #trains the data based on both training subsets of input and output
    y_pred_baseMLP = baseMLP.predict(x_test) #provides an output (species) prediction based on a input test subset
    accuracy, macro_f1, weighted_f1 = evaluate(y_test, y_pred_baseMLP)
    #5B)
    confusionMatrix(name,y_test, y_pred_baseMLP)
    #5C)
    classificationReport(name,y_test, y_pred_baseMLP)
    #5D)
    report_performance(name ,"Base-MLP", accuracy, macro_f1, weighted_f1)
    print('Base-MLP Performance for Penguins:')
    #Comparing true results with prediction
    print(classification_report(y_test, y_pred_baseMLP, zero_division=1))
    return accuracy, macro_f1, weighted_f1

def topMLP(name:str, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    #Instructions on how to conduct search (Mapping)
    topMLPparams = {
    'activation': ['logistic', 'tanh', 'relu'], #activation functions
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)], #Network architectures
    'solver': ['adam', 'sgd'] #Solver for weight distribution
    }
    #Use GridSearchCV to perform an exhaustive search using acuracy as the scoring.
    topMLP = GridSearchCV(MLPClassifier(max_iter=1000), topMLPparams, scoring='accuracy') #setting max_iter for convergence warnings
    #To train the model
    topMLP.fit(x_train, y_train)
    print('Top-MLP Best Parameters:', topMLP.best_params_)
    y_pred_topMLP = topMLP.predict(x_test)
    accuracy, macro_f1, weighted_f1 = evaluate(y_test, y_pred_topMLP)
    #5B)
    confusionMatrix(name,y_test, y_pred_topMLP)
    #5C)
    classificationReport(name,y_test, y_pred_topMLP)
    #5D)
    report_performance(name ,"Top-MLP", accuracy, macro_f1, weighted_f1)
    print('Top-MLP Performance for Penguins:')
    print(classification_report(y_test, y_pred_topMLP, zero_division=1))
    return accuracy, macro_f1, weighted_f1
        
def penguinsSteps(penguin_data, user_choice):

    if user_choice == 1:
        #1: Convert Categorical Features for Penguins into 1 hot vector
        penguins = pd.get_dummies(penguin_data, columns=['island', 'sex'], drop_first=True) #drop first used to drop the species column
    elif user_choice == 2:
        #2: manual categorizations
        penguins = penguin_data
        penguins['species'] = penguins['species'].astype('category').cat.codes
        penguins['island'] = penguins['island'].astype('category').cat.codes
        penguins['sex'] = penguins['sex'].astype('category').cat.codes

    #2:
    plot_distribution(penguins, "species", "penguins-classes.png")
    
    #3: split data
    x = penguins.drop('species', axis=1)
    y = penguins['species']
    with open('penguins-performance.txt', 'w') as file:
        file.write("Base-DT Evaluation Results\n")
    accuracy_metrics = {'Base-DT': [], 'Top-DT': [], 'Base-MLP': [], 'Top-MLP': []}
    macro_metrics = {'Base-DT': [], 'Top-DT': [], 'Base-MLP': [], 'Top-MLP': []}
    weighted_metrics = {'Base-DT': [], 'Top-DT': [], 'Base-MLP': [], 'Top-MLP': []}
    
    #4a) Base-DT for Penguins
    for i in range(RANGE):
        #5A)
        with open(f'penguins-performance.txt', 'a') as file:
            file.write("--------------------------------------------------\n")
            file.write(f"Base-DT Model -- Iteration {i}\n")
        accuracy, macro_f1, weighted_f1 = baseDT("penguins", x, y)
        accuracy_metrics['Base-DT'].append(accuracy)
        macro_metrics['Base-DT'].append(macro_f1)
        weighted_metrics['Base-DT'].append(weighted_f1)
    #6 Appended Averages
    appendperformance("penguins","Base-DT", accuracy_metrics)
    appendperformance("penguins","Base-DT", macro_metrics)
    appendperformance("penguins","Base-DT", weighted_metrics)
    
    #4b) Top-DT for Penguins
    for i in range(RANGE):
        #5A)
        with open(f'penguins-performance.txt', 'a') as file:
            file.write("--------------------------------------------------\n")
            file.write(f"Top-DT Model -- Iteration {i}\n")
        accuracy, macro_f1, weighted_f1 = topDT("penguins", x, y)
        accuracy_metrics['Top-DT'].append(accuracy)
        macro_metrics['Top-DT'].append(macro_f1)
        weighted_metrics['Top-DT'].append(weighted_f1)
    #6 Appended Averages
    appendperformance("penguins","Top-DT", accuracy_metrics)
    appendperformance("penguins","Top-DT", macro_metrics)
    appendperformance("penguins","Top-DT", weighted_metrics)
    
    #4c) Base MLP for Penguins    
    for i in range(RANGE):
        #5A)
        with open(f'penguins-performance.txt', 'a') as file:
            file.write("--------------------------------------------------\n")
            file.write(f"Base-MLP Model -- Iteration {i}\n")
        accuracy, macro_f1, weighted_f1 = baseMLP("penguins", x, y)
        accuracy_metrics['Base-MLP'].append(accuracy)
        macro_metrics['Base-MLP'].append(macro_f1)
        weighted_metrics['Base-MLP'].append(weighted_f1)
    #6 Appended Averages
    appendperformance("penguins","Base-MLP", accuracy_metrics)
    appendperformance("penguins","Base-MLP", macro_metrics)
    appendperformance("penguins","Base-MLP", weighted_metrics)
    
    #4d) Top MLP for Penguins
    for i in range(RANGE):
        #5A)
        with open(f'penguins-performance.txt', 'a') as file:
            file.write("--------------------------------------------------\n")
            file.write(f"Top-MLP Model -- Iteration {i}\n")
        accuracy, macro_f1, weighted_f1 = topMLP("penguins", x, y)
        accuracy_metrics['Top-MLP'].append(accuracy)
        macro_metrics['Top-MLP'].append(macro_f1)
        weighted_metrics['Top-MLP'].append(weighted_f1)
    #6 Appended Averages
    appendperformance("penguins","Top-MLP", accuracy_metrics)
    appendperformance("penguins","Top-MLP", macro_metrics)
    appendperformance("penguins","Top-MLP", weighted_metrics)
    

def abaloneSteps(abalone):
    #2:
    plot_distribution(abalone, "Type", "abalone-classes.png")
    
    #3 split data
    x = abalone.drop('Type', axis=1)
    y = abalone['Type']
    accuracy_metrics = {'Base-DT': [], 'Top-DT': [], 'Base-MLP': [], 'Top-MLP': []}
    macro_metrics = {'Base-DT': [], 'Top-DT': [], 'Base-MLP': [], 'Top-MLP': []}
    weighted_metrics = {'Base-DT': [], 'Top-DT': [], 'Base-MLP': [], 'Top-MLP': []}
    
    #4a) Base-DT for Abalone
    for i in range(RANGE):
        #5A)
        with open(f'abalone-performance.txt', 'a') as file:
            file.write("--------------------------------------------------\n")
            file.write(f"Base-DT Model -- Iteration {i}\n")
        accuracy, macro_f1, weighted_f1 = baseDT("abalone", x, y)
        accuracy_metrics['Base-DT'].append(accuracy)
        macro_metrics['Base-DT'].append(macro_f1)
        weighted_metrics['Base-DT'].append(weighted_f1)
    #6 Appended Averages
    appendperformance("abalone","Base-DT", accuracy_metrics)
    appendperformance("abalone","Base-DT", macro_metrics)
    appendperformance("abalone","Base-DT", weighted_metrics)
    
    #4b) Top-DT for Abalone
    for i in range(RANGE):
        #5A)
        with open(f'abalone-performance.txt', 'a') as file:
            file.write("--------------------------------------------------\n")
            file.write(f"Top-DT Model -- Iteration {i}\n")
        accuracy, macro_f1, weighted_f1 = topDT("abalone", x, y)
        accuracy_metrics['Top-DT'].append(accuracy)
        macro_metrics['Top-DT'].append(macro_f1)
        weighted_metrics['Top-DT'].append(weighted_f1)
    #6 Appended Averages
    appendperformance("abalone","Top-DT", accuracy_metrics)
    appendperformance("abalone","Top-DT", macro_metrics)
    appendperformance("abalone","Top-DT", weighted_metrics)
    
    #4c) Base MLP for Abalone    
    for i in range(RANGE):
        #5A)
        with open(f'abalone-performance.txt', 'a') as file:
            file.write("--------------------------------------------------\n")
            file.write(f"Base-MLP Model -- Iteration {i}\n")
        accuracy, macro_f1, weighted_f1 = baseMLP("abalone", x, y)
        accuracy_metrics['Base-MLP'].append(accuracy)
        macro_metrics['Base-MLP'].append(macro_f1)
        weighted_metrics['Base-MLP'].append(weighted_f1)
    #6 Appended Averages
    appendperformance("abalone","Base-MLP", accuracy_metrics)
    appendperformance("abalone","Base-MLP", macro_metrics)
    appendperformance("abalone","Base-MLP", weighted_metrics)
    
    #4d) Top MLP for Abalone
    for i in range(RANGE):
        #5A)
        with open(f'abalone-performance.txt', 'a') as file:
            file.write("--------------------------------------------------\n")
            file.write(f"Top-MLP Model -- Iteration {i}\n")
        accuracy, macro_f1, weighted_f1 = topMLP("abalone", x, y)
        accuracy_metrics['Top-MLP'].append(accuracy)
        macro_metrics['Top-MLP'].append(macro_f1)
        weighted_metrics['Top-MLP'].append(weighted_f1)
    #6 Appended Averages
    appendperformance("abalone","Top-MLP", accuracy_metrics)
    appendperformance("abalone","Top-MLP", macro_metrics)
    appendperformance("abalone","Top-MLP", weighted_metrics)

  
def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))  # Get the directory/filepath of the script
    abalone_file_path = os.path.join(script_directory, "abalone.csv")
    penguins_file_path = os.path.join(script_directory, "penguins.csv")
    penguin_data = pd.read_csv(penguins_file_path)
    abalone_data = pd.read_csv(abalone_file_path)
    
    while(True):
        user_choice = string_to_int(input("Please select one of the options below to check a file.\n(1) abalone.csv\n(2) penguins.csv\n"))
        if user_choice == 1:
            print("abalone.csv has been selected")
            abaloneSteps(abalone_data)
            break
        elif user_choice == 2:
            print("penguins.csv has been selected")
            user_choice2 = string_to_int(input("Select which way the Penguins database will converted:\n(1) 1-hot vector \n(2) categorize manually\n"))
            if user_choice2 == 1:
                print("1-hot vector will be used")
                penguinsSteps(penguin_data, user_choice2)
                break
            elif user_choice2 == 2:
                print("Manual conversion will be used")
                penguinsSteps(penguin_data, user_choice2)
                break
            else:
                print("[Error]: Invalide option has been selected.")
        else:
            print("[Error]: Invalide option has been selected.")

if __name__ == "__main__":
    main()