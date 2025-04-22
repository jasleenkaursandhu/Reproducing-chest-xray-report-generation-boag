import csv
from time import strftime, gmtime
import os

# Define the base path - ADJUST THIS TO YOUR ACTUAL PATH
base_path = "/Users/jasleensandhu/Desktop/CS598DLH/"

# Process 3-gram.tsv
input_file = os.path.join(base_path, "3-gram.tsv")
output_file = os.path.join(base_path, "3gram_headerless.csv")

# Create a dictionary to store the reports
ngram_reports = {}

# Read the input file
with open(input_file, "r") as f:
    # Check if there's a header by reading the first line
    first_line = f.readline().strip()
    has_header = 'dicom_id' in first_line and '\t' in first_line
    
    # If there's a header, we've already consumed it, if not we need to process the line
    if not has_header:
        # Process the first line as it contains data
        parts = first_line.split('\t')
        if len(parts) >= 2:
            dicom_id = parts[0]
            text = '\t'.join(parts[1:])  # In case there are multiple tabs in the text
            ngram_reports[dicom_id] = text
    
    # Process the rest of the lines
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            dicom_id = parts[0]
            text = '\t'.join(parts[1:])  # In case there are multiple tabs in the text
            ngram_reports[dicom_id] = text

# Now write the reports to the output file in the required format
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    # No header (headerless as required)
    for dicom_id, text in sorted(ngram_reports.items()):
        # Only include the text as a single column
        # Also replace commas with escaped commas for CSV format
        text = text.replace(",", "\\,")
        writer.writerow([text])

print(f"Processed {len(ngram_reports)} n-gram reports")
print(f"Output written to {output_file}")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))