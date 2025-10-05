import csv

def filter_rows(filename):
    filtered_rows = []
    prev_row = None
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if prev_row is not None and row[0] != prev_row[0]:
                filtered_rows.append(prev_row)
            prev_row = row
    if prev_row is not None and prev_row not in filtered_rows:
        filtered_rows.append(prev_row)
    return filtered_rows
