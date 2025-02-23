import os
import subprocess

def process_in_batches(theme, batch_size, first, compiler, python_script):
    """Processes files in batches, skipping files before 'first' and avoiding duplicates."""
    num_files = len(os.listdir(f'Sounds/{theme}'))
    
    if num_files < first:
        print(f"No files to process for theme '{theme}'. 'first' ({first}) exceeds the number of files ({num_files}).")
        return

    start = max(0, first)  # Ensure start is not negative and begins at 'first'

    for i in range(start // batch_size, (num_files + batch_size - 1) // batch_size):  # Corrected loop range
        start = i * batch_size
        end = min((i + 1) * batch_size, num_files) # Ensure end doesn't exceed num_files

        if start < num_files: # Only process if there are files in the range.
            print(start, end)
            subprocess.Popen([compiler, python_script, str(theme), str(start), str(end)])

batch_size = 400
first = 0
theme = 'fireworks_concatenate'
python_script = "spectrograms.py"
compiler = r"C:\anaconda3\envs\protect_of_terrorist_attacks\python.exe" 

process_in_batches(theme, batch_size, first, compiler, python_script) 

# for i in range(len(os.listdir(f'Sounds/{theme}'))//batch_size):
#     num_files = len(os.listdir(f'Sounds/{theme}'))
#     for i in range(num_files // batch_size):
#         start = i * batch_size
#         end = (i + 1) * batch_size
#         if end >= first:
#             print(start, end)
#             subprocess.Popen([compiler, python_script, str(theme), str(start), str(end)])
    
#     remaining_files = num_files % batch_size
#     if remaining_files > 0 and num_files >= first:
#         start = num_files - remaining_files
#         end = num_files
#         print(start, end)
#         subprocess.Popen([compiler, python_script, str(theme), str(start), str(end)])