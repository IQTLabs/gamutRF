import argparse
import logging
import os
import sys
import time
import tarfile
import gzip
import subprocess
import sysrsync


def check_tld(top_dir, args):
    if not os.path.exists(top_dir):
        print(f"Top-level directory '{top_dir}' does not exist.")
        return []

    if not os.path.isdir(top_dir):
        print(f"'{top_dir}' is not a directory.")
        return []

    valid_folders = []
    current_time = time.time()

    print(
        f"Folders inside '{top_dir}' that are more than {args.threshold_seconds} seconds old:"
    )

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
    # subdirectories = [subdir for subdir in os.listdir(top_dir)
    #                  if os.path.isdir(os.path.join(top_dir, subdir))]

    tar_filenames = []

    for dir_path in dir_paths:
        # Create a tar file object
        if args.compress:
            tar_filename = f"{dir_path}.tar.gz"
            file_mode = "w:gz"
        else:
            tar_filename = f"{dir_path}.tar"
            file_mode = "w"

        with tarfile.open(tar_filename, file_mode) as tar_file:
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
    base_filename = os.path.basename(filename)
    exported_filepath = os.path.join(export_path, base_filename)

    if args.export_ssh_host and args.export_ssh_key:
        sysrsync.run(
            source=filename,
            destination=export_path,
            destination_ssh=args.export_ssh_host,
            private_key=args.export_ssh_key,
            options=["-r", "-P", "-u"],
            sync_source_contents=False,
        )
    elif args.export_ssh_host and not args.export_ssh_key:
        sysrsync.run(
            source=filename,
            destination=export_path,
            destination_ssh=args.export_ssh_host,
            options=["-r", "-P", "-u"],
            sync_source_contents=False,
        )
    else:
        print(sysrsync.get_rsync_command(source=filename, destination=export_path))
        sysrsync.run(
            source=filename,
            destination=export_path,
            options=["-r", "-P", "-u"],
            sync_source_contents=False,
        )

    print(f"Exported {filename} to {exported_filepath}")

    return exported_filepath


def argument_parser():
    parser = argparse.ArgumentParser(
        description="tar and compress recording directories"
    )
    parser.add_argument(
        "dir", default="", type=str, help="Top level directory containing recordings"
    )
    parser.add_argument(
        "--compress",
        dest="compress",
        action="store_true",
        default=False,
        help="compress (gzip) directories",
    )
    parser.add_argument(
        "--delete",
        dest="delete",
        action="store_true",
        default=False,
        help="delete after compressing",
    )
    parser.add_argument(
        "--threshold_seconds",
        dest="threshold_seconds",
        type=int,
        default=300,
        help="modtime threshold for folders to be considered, must be older than threshold_seconds",
    )
    parser.add_argument(
        "--export_path",
        dest="export_path",
        type=str,
        default=None,
        help="path to export to after processing",
    )
    parser.add_argument(
        "--export_ssh_host",
        dest="export_ssh_host",
        type=str,
        default=None,
        help="host for external export using ssh",
    )
    parser.add_argument(
        "--export_ssh_key",
        dest="export_ssh_key",
        type=str,
        default=None,
        help="if using external_ssh_host a ssh key can be specified",
    )

    return parser


def check_args(args):
    if args.export_ssh_host and not args.export_ssh_key:
        return "If using export_ssh_key please include export_ssh_host"
    return ""


def main():
    args = argument_parser().parse_args()
    results = check_args(args)
    if results:
        print(results)
        sys.exit(1)

    # Check for valid subdirs
    valid_folders = check_tld(args.dir, args)

    # Process subdirs
    if valid_folders:
        tarred_filenames = tar_directories(valid_folders, args)
        print("Tarred files:")
        for filename in tarred_filenames:
            print(filename)
        print("...finished processing files")
        if args.export_path != None:
            print(f"Exporting tar files to {args.export_path}")
            exported_filepath = export_to_path(args.dir, args.export_path, args)
            print(f"...done exporting to {args.export_path}")
    else:
        print("No valid folders found.")


if __name__ == "__main__":
    main()
