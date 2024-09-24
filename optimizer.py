import GSA as gsa
import benchmarks as benchmarks
import optiFeatures as optiFeatures

import csv
import numpy
import time
import argparse
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


def selector(func_details, args, df, clf, metric):
    # Unpack function details: function name, lower and upper bounds, dimensionality
    function_name = func_details[0]  # Name of fitness function
    lb = func_details[1]  # Lower bound for feature values
    ub = func_details[2]  # Upper bound for feature values
    dim = func_details[3]  # Dimensionality of the search space (number of features)
    
    # Call GSA function to optimize the feature selection based on benchmark fitness function
    gsa_results = gsa.GSA(getattr(benchmarks, function_name), lb, ub, dim, args, df, clf, metric)  
    
    # Apply the best solution found by GSA to the classification task
    classification_results = optiFeatures.optiFeats(gsa_results.gBest, df, clf)  
    
    # Store the classification results
    gsa_results.testAcc = classification_results[0]
    gsa_results.testF1 = classification_results[1]
    gsa_results.testTP = classification_results[2]
    gsa_results.testFN = classification_results[3]
    gsa_results.testFP = classification_results[4]
    gsa_results.testTN = classification_results[5] 
    
    return gsa_results

def collect_arguments():
    """
    Collect all possible arguments using argparse for the feature selection process.
    
    Returns:
    - args: Parsed command-line arguments.
    """
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Feature Selection with GSA and customizable model and metric.")
    
    # Add arguments
    parser.add_argument('--classifier', type=str, default='SVM', choices=['SVM', 'RF', 'LR', 'DT', 'GB', 'KNN'], 
                        help='Choose the classifier model to use (default: SVC)')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'f1'],
                        help='Choose the performance metric (default: acc)')
    parser.add_argument('--early_stop', type=bool, default=True,
                        help='Stop early when performance saturates (default: True)')
    parser.add_argument('--patience', type=int, default=10,
                        help='After how many iterations with no improvement should the training stop (default: 10)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset file (CSV format)')
    parser.add_argument('--ignore_first', type=bool, default=True,
                        help='Ignore the first column of the csv, in case it is an id (default: True)')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Proportion of the dataset to include in the test split (default: 0.30)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold to select features based on GSA results (default: 0.5)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of independent runs for the GSA algorithm (default: 1)')
    parser.add_argument('--pop_size', type=int, default=5,
                        help='Population size for the GSA optimization process (default: 5)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations for GSA (default: 10)')
    parser.add_argument('--export', type=bool, default=True,
                        help='Export results to a CSV file (default: True)')
    parser.add_argument('--export_name', type=str, default="experiment",
                        help='Filename to export the results (default: exp)')
    parser.add_argument('--fitness', type=str, default="F1",
                        help='Which fitness function to use, they can be defined in benchmarks.py (default: F1)')
    

    # Parse the arguments
    args = parser.parse_args()
    
    return args

if __name__ == '__main__': 
    args = collect_arguments()

    print(f'{args = }')

    if args.classifier == 'SVM':
        model = SVC()
    elif args.classifier == 'RF':
        model = RandomForestClassifier()
    elif args.classifier == 'LR':
        model = LogisticRegression(solver='saga', max_iter=500)
    elif args.classifier == 'DT':
        model = DecisionTreeClassifier()
    elif args.classifier == 'GB':
        model = GradientBoostingClassifier()
    elif args.classifier == 'KNN':
        model = KNeighborsClassifier()

    if args.metric == 'acc':
        metric = accuracy_score
    elif args.metric == 'f1':
        metric = f1_score

    ExportToFile = './experiments/' + args.export_name + args.classifier + '.csv'

    # Track whether at least one iteration is performed
    atLeastOneIteration = False

    # Number of columns to not consider in evaluation
    ignoredCols = 2 if args.ignore_first else 1

    # CSV header to record convergence metrics for each iteration
    CnvgHeader = ["Iter" + str(i + 1) for i in range(args.iterations)]

    df = pd.read_csv(args.dataset)

    # Run the optimization for the defined number of independent runs
    for k in range(args.runs):
        
        # Define function details: fitness function and feature space bounds
        func_details = ["F1", 0, 1, (len(df.columns) - ignoredCols)]  # ADJUST THIS TO MY DATA -2 because Iris has two categorical labels
        
        # Run GSA
        x = selector(func_details, args, df, model, metric)
        
        # Export results into a CSV
        if args.export:
            with open(ExportToFile, 'a') as out:
                writer = csv.writer(out, delimiter=',')
                
                # Write the header once if it's the first iteration
                if not atLeastOneIteration:
                    header = numpy.concatenate([["Optimizer", "Dataset", "objfname", "Model", "Metric", "Experiment", "startTime", "EndTime", "ExecutionTime", "testAcc", "testF1", "testTP", "testFN", "testFP", "testTN"], CnvgHeader])
                    writer.writerow(header)
                
                # Record results and convergence data
                data = numpy.concatenate([[x.Algorithm, args.dataset, x.objectivefunc, args.classifier, args.metric, k + 1, x.startTime, x.endTime, x.executionTime, x.testAcc, x.testF1, x.testTP, x.testFN, x.testFP, x.testTN], x.convergence])
                writer.writerow(data)
            
        atLeastOneIteration = True  # Mark that at least one iteration was run
