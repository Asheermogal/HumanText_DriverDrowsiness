import tarfile


def extract_tar_gz(file_path, output_path):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=output_path)


file_path = r"C:\Users\ashee\Downloads\IIIT5K-Word_V3.0.tar.gz"  # replace with your file path
output_path = r'E:\Data Science Project\Humantext Recognition\Datasets'  # replace with the path where you want to extract the files
extract_tar_gz(file_path, output_path)
