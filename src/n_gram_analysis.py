import os
import re
import csv
import argparse


def search_and_save_patterns(dataset_dir, patterns_file, output_csv_file):
    # Initialize a list to store the results
    results = []

    # Step 1: Read regex patterns from a file
    with open(patterns_file, 'r') as patterns_file:
        regex_patterns = patterns_file.read().splitlines()

    # Step 2: Load and search text files
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as text_file:
                    text_data = text_file.read()

                # Step 3: Filter out single-word patterns
                filtered_patterns = [pattern for pattern in regex_patterns if len(pattern.split()) > 1]

                # Step 4: Replace one word with a wildcard and search for matches
                for pattern in filtered_patterns:
                    # Split the pattern into words
                    words = pattern.split()
                    for i in range(len(words)):
                        # Create a copy of the original pattern
                        modified_pattern = pattern[:]
                        # Replace one word with a wildcard (e.g., \w+ for any word)
                        modified_pattern = re.sub(r'\b' + words[i] + r'\b', r'\w+', modified_pattern)
                        # Compile the modified pattern
                        compiled_pattern = re.compile(modified_pattern, re.IGNORECASE)
                        # Search for matches
                        if compiled_pattern.search(text_data):
                            # If a match is found, add the original pattern and modified pattern to the results
                            results.append({'Original Pattern': pattern, 'Modified Pattern': modified_pattern})
                            break  # No need to continue searching for this pattern

    # Step 5: Save the results to a CSV file
    with open(output_csv_file, 'w', newline='') as csv_file:
        fieldnames = ['Original Pattern', 'Modified Pattern']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Search for and save patterns in clinical text data.")
    parser.add_argument("dataset_dir",
                        help="Directory containing text files to search",
                        default="datasets/cardio-oncology/data"
                        )

    parser.add_argument("patterns_file", help="File containing regex patterns")
    parser.add_argument("output_csv_file", help="Output CSV file to save results")

    args = parser.parse_args()

    search_and_save_patterns(args.dataset_dir, args.patterns_file, args.output_csv_file)


if __name__ == "__main__":
    main()
