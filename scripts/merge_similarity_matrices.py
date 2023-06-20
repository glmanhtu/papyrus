import argparse

import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--similarity_files', type=str, help='Specify all similarity matrices to merge', required=True,
                    nargs='+')
parser.add_argument('--output_file', type=str, help='Name of the output file', required=True)

args = parser.parse_args()

dataframes = {}
categories = set([])
for file in args.similarity_files:
    df = pd.read_csv(file, index_col=0)
    name_mapping = {}
    for column in set(df.columns):
        name_mapping[column] = column.replace('_COLV', '').replace('_COLR', '').replace('_IRR', '').replace('_IRV', '')
    df = df.rename(columns=name_mapping, index=name_mapping)
    dataframes[file] = df
    categories = set(df.columns) if len(categories) == 0 else set(categories & set(df.columns))

categories = sorted(list(categories))
for df in dataframes:
    # Use only the common categories between similarity matrix files
    dataframes[df] = dataframes[df].loc[categories, categories]

# Merge the dataframes together by taking the average of the values for each category
merged_df = pd.concat(dataframes.values()).groupby(level=0).mean()
# for category in categories:
#     avg = []
#     for dataframe in dataframes:
#         avg.append(dataframes[dataframe][category][category])
#     avg_val = sum(avg) / len(avg)
#     assert avg_val == merged_df[category][category]

# Write the merged dataframe to a new CSV file
merged_df.to_csv(args.output_file)
