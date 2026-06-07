import pandas as pd
import os

def reduce_csv_size(input_file, output_file, target_size_mb):
    """
    Reduces the number of rows in a CSV file to meet the target file size.
    
    :param input_file: Path to the input CSV file.
    :param output_file: Path to save the reduced CSV file.
    :param target_size_mb: Desired file size in MB.
    """
    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return

    # Estimate the size of the original file
    original_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    print(f"Original file size: {original_size_mb:.2f} MB")

    # Read the entire CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Calculate the reduction factor
    reduction_factor = target_size_mb / original_size_mb

    if reduction_factor >= 1:
        print("The file is already smaller than or equal to the target size.")
        df.to_csv(output_file, index=False)  # Save the original file as output
        print(f"The original file has been saved as '{output_file}'.")
        return

    # Determine the number of rows to keep
    total_rows = len(df)
    rows_to_keep = int(total_rows * reduction_factor)

    # Reduce the DataFrame
    reduced_df = df.iloc[:rows_to_keep]

    # Save the reduced DataFrame to a new CSV file
    reduced_df.to_csv(output_file, index=False)

    # Confirm the new file size
    new_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Reduced file size: {new_size_mb:.2f} MB")
    print(f"Rows reduced from {total_rows} to {rows_to_keep}")
    
if __name__ == "__main__":
    # Replace these paths with your input and output files
    input_csv_path = "IoT_Network_Intrusion_Dataset.csv"
    output_csv_path = "Small_IoT_Network_Intrusion_Dataset.csv"
    
    # Target size in MB
    target_file_size_mb = 25
    
    # Ensure output directory exists (if applicable)
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory for output: {output_dir}")
    
    reduce_csv_size(input_csv_path, output_csv_path, target_file_size_mb)
