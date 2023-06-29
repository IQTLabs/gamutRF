import argparse
import concurrent.futures
import os
import time
import tarfile
import gzip
import subprocess

def check_tld(top_dir, args):

    if not os.path.exists(top_dir):
        print(f"Top-level directory '{top_dir}' does not exist.")
        return []

    if not os.path.isdir(top_dir):
        print(f"'{top_dir}' is not a directory.")
        return []

    valid_folders = []
    current_time = time.time()

    print(f"Folders inside '{top_dir}' that are more than {args.threshold_seconds} seconds old:")

    for dir in os.listdir(top_dir):
        dir_path = os.path.join(top_dir, dir)
        if os.path.isdir(dir_path):
            folder_mtime = os.path.getmtime(dir_path)
            if current_time - folder_mtime > args.threshold_seconds:
                print(f"Folder: {dir}")
                valid_folders.append(dir_path)

    return valid_folders

def tar_directories(dir_paths, args):

    # Get a list of subdirectories within the top-level directory
    #subdirectories = [subdir for subdir in os.listdir(top_dir)
    #                  if os.path.isdir(os.path.join(top_dir, subdir))]

    tar_filenames = []
    
    for dir_path in dir_paths:

        # Create a tar file object
        if args.compress:
            tar_filename = f"{dir_path}.tar.gz"
            tar_file = tarfile.open(tar_filename, "w:gz")
        else:
            tar_filename = f"{dir_path}.tar"
            tar_file = tarfile.open(tar_filename, "w")
            
        
        # Add all files in the directory to the tar file
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                tar_file.add(file_path)
        
        # Close the tar file
        tar_file.close()
        
        tar_filenames.append(tar_filename)
        
        # Delete if --delete
        if args.delete:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
            os.rmdir(dir_path)
    
    return tar_filenames

def export_to_path(filename, export_path, args):
    
    base_filename=os.path.basename(filename)
    exported_filepath=os.path.join(export_path, base_filename)

    rsync_cmd=f'rsync -rqhPu {filename} {export_path}'
    p=subprocess.Popen(rsync_cmd.split())

    print(f'Exported {filename} to {exported_filepath}')

    return exported_filepath

def argument_parser():

    parser = argparse.ArgumentParser(description="tar and compress recording directories")
    parser.add_argument(
        "dir", default="", type=str, help="Top level directory containing recordings"
    )
    parser.add_argument("--compress", dest="compress", action="store_true", help="compress (gzip) directories")
    parser.add_argument("--delete", dest="delete", action="store_true", help="delete after compressing")
    parser.add_argument("--threshold_seconds", dest="threshold_seconds", type=int, help="delete after compressing")
    parser.add_argument("--export", dest="export_path", type=str, help="internal or external path to export to after processing")
    parser.set_defaults(dir="", dont_compress=False, delete=False, threshold_seconds=300, export_path="")

    return parser


def main():
    args = argument_parser().parse_args()

    # Check for valid subdirs
    valid_folders = check_tld(args.dir, args)

    # Process subdirs
    if valid_folders:
        tarred_filenames = tar_directories(valid_folders, args)
        print("Tarred files:")
        for filename in tarred_filenames:
            print(filename)
        print('...finished processing files')
        if args.export_path != "":
            print(f'Exporting tar files to {args.export_path}')
            exported_filepath = export_to_path(args.dir, args.export_path, args)
            print(f'...done exporting to {args.export_path}')
    else:
        print("No valid folders found.")

if __name__ == "__main__":
    main()

