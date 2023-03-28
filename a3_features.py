#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# Whatever other imports you need

def is_signature(line):
    CONDITIONS = [
        line == '-- \n',
        line == '--\n',
        line.startswith('Best regards'),
        line.startswith('Best,'),
        line.startswith('Kind regards,'),
        line.startswith('Kind,'),
        line.startswith('Sincerely,'),
        line.startswith(' -----Original Message-----'),
    ]
    # If any of CONDITIONS is true, return True
    return any(CONDITIONS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    emails = []
    labels = []
    paths = []
    for person in os.listdir(args.inputdir):
        person_directory = args.inputdir + "/" + person
        for email in os.listdir(person_directory):
            file_path  = person_directory + "/" + email
            with open(file_path, "r") as f:
                lines = f.readlines()
                # Remove headers. The last line of the header is usually the first empty line
                for i, line in enumerate(lines):
                    if line == "\n":
                        lines = lines[i+1:]
                        break
                # Remove signature lines. Use is_signature() to check if a line is a signature line
                for i, line in enumerate(lines):
                    if is_signature(line):
                        lines = lines[:i]
                        break
                email = " ".join(lines)
                emails.append(email)
                labels.append(person)
                paths.append(file_path)

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Split the data into training and test sets and vectorize the data
    X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=args.testsize / 100, random_state=42)
    vectorizer = TfidfVectorizer(max_features=args.dims)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    y_train_vec = vectorizer.transform(y_train)
    y_test_vec = vectorizer.transform(y_test)
    # Store all data in a single DataFrame
    df_train = pd.DataFrame(X_train_vec.toarray(), columns=vectorizer.get_feature_names())
    df_test = pd.DataFrame(X_test_vec.toarray(), columns=vectorizer.get_feature_names())
    df_train['label'] = y_train
    df_test['label'] = y_test
    df_train['train'] = True
    df_test['train'] = False
    df = pd.concat([df_train, df_test])
    
    print("Writing to {}...".format(args.outputfile))
    df.to_csv(args.outputfile, index=False)

    print("Done!")
    
