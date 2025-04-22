import csv
from time import strftime, gmtime
import os

# Define the base path
base_path = "/Users/jasleensandhu/Desktop/CS598DLH/"

# Process random.tsv
input_file = os.path.join(base_path, "random.tsv")
output_file = os.path.join(base_path, "random_headerless.csv")

# Create a dictionary to store the reports
random_reports = {}

# Read the input file
with open(input_file, "r") as f:
    # Check if there's a header
    first_line = f.readline().strip()
    has_header = 'dicom_id' in first_line and '\t' in first_line
    
    # Process header or first line
    if not has_header:
        parts = first_line.split('\t')
        if len(parts) >= 2:
            dicom_id = parts[0]
            text = '\t'.join(parts[1:])
            random_reports[dicom_id] = text
    
    # Process remaining lines
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            dicom_id = parts[0]
            text = '\t'.join(parts[1:])
            random_reports[dicom_id] = text

# Write reports to output file
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    for dicom_id, text in sorted(random_reports.items()):
        text = text.replace(",", "\\,")
        writer.writerow([text])

print(f"Processed {len(random_reports)} random reports")
print(f"Output written to {output_file}")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))