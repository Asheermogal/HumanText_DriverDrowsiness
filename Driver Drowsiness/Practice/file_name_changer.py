import os

active_subjects_path = r'E:\Data Science Project\Driver Drowsiness\Datasets\FaceImages\Active Subjects'
fatigue_subjects_path = r'E:\Data Science Project\Driver Drowsiness\Datasets\FaceImages\Fatigue Subjects'

# Function to rename files in a directory with a proper sequence
def rename_files(directory_path):
    file_list = os.listdir(directory_path)
    sorted_file_list = sorted(file_list)

    for i, file_name in enumerate(sorted_file_list):
        file_path = os.path.join(directory_path, file_name)
        new_file_name = f'Pic_{i+1}.jpg'  # Set the new file name format
        new_file_path = os.path.join(directory_path, new_file_name)

        # Check if the new file name already exists
        if not os.path.exists(new_file_path):
            os.rename(file_path, new_file_path)
        else:
            print(f"Skipped renaming '{file_name}' due to a conflict.")

# Rename files in the Active Subjects folder
rename_files(active_subjects_path)

# Rename files in the Fatigue Subjects folder
rename_files(fatigue_subjects_path)
