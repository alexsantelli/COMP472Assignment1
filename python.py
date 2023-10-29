import os
import sys
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
from sklearn.tree import DecisionTreeClassifier #https:/https://github.com/microsoft/pyright/blob/main/docs/configuration.md#reportMissingModuleSource/scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier #https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
from sklearn.model_selection import GridSearchCV #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html


def string_to_int(str) -> int:
    try:
        output = int(str)
    except ValueError:
        print("invalid Input. Enter Integer value\n")
        return 5
    return output

def plot_class_distribution(data, dataset_name):
    class_counts = data['target_column'].value_counts()
    class_percentage = class_counts / len(data)
    plt.bar(class_counts.index, class_percentage)
    plt.xlabel("Class")
    plt.ylabel("Percentage")
    plt.title(f"{dataset_name} Class Distribution")
    plt.savefig(f"{dataset_name}-classes.gif")
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
                read_csv(abalone_data)
                break
            elif user_choice == 2:
                print("penguins.csv has been selected")
                #read_csv(penguins_file_path)
                break
            elif user_choice == 3:
                while(True):
                    user_input_custom = input("Custom file has been selects\nPlease enter the file path and name that you want to review")
                    if user_input_custom: 
                        #read_csv(user_input)
                        break
                    else:
                        print("[Error]: User input is empty. Please enter a valid file.")    
            else:
                print("[Error]: Invalide option has been selected.")
        else:
            print("[Error]: User input is empty. Please enter a valid file.")

if __name__ == "__main__":
    main()