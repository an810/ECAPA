import csv
import pandas as pd

# File paths for the input and output files
input_file = "eval_asv.txt"
output_file = "eval_asv.csv"


# Open the input TSV file for reading and the output CSV file for writing
# with open(input_file, 'r') as tsvfile, open(output_file, 'w', newline='') as csvfile:
#     # Read each line from the input TSV file and split it by tabs
#     for line in tsvfile:
#         values = line.strip().split('\t')
        
#         # Write the split values to the CSV file, using a comma as the delimiter
#         csvfile.write(','.join(values) + '\n')

# print(f"TSV file '{input_file}' has been converted to CSV file '{output_file}' with a comma delimiter.")


h = pd.read_csv(output_file, sep='\t', header = None)
print(h)