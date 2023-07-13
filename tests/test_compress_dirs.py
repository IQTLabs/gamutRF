import time
import os
import pytest
import unittest

from gamutrf.compress_dirs import check_tld
from gamutrf.compress_dirs import tar_directories
from gamutrf.compress_dirs import argument_parser 
from gamutrf.compress_dirs import export_to_path

TESTDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/gamutrf_test_dir")

class FakeArgs:
    def __init__(
        self,
        dir=None,
        delete=False,
        compress=False,
        threshold_seconds=300,
        export_path=None
    ):
        self.dir=dir
        self.delete=delete
        self.compress=compress
        self.threshold_seconds=threshold_seconds
        self.threshold_path=threshold_seconds


def test_argument_parser():
    argument_parser()

def test_check_valid_tld():
    args=FakeArgs(TESTDIR, False, True, 300, "")
    folders=check_tld(TESTDIR, args)
    assert("00001" in folders)
    assert("00002" in folders)
    assert("00003" in folders)
    assert("scan.csv" not in folders)

def test_check_invalid_tld():
    INVALID_TESTDIR=TESTDIR+"_invalid"
    args=FakeArgs(INVALID_TESTDIR, False, True, 300, "")
    folders=check_tld(TESTDIR, args)
    assert(folders.__sizeof__ == 0)

def test_tar_directories():
    #Test without compression
    args=FakeArgs(TESTDIR, False, False, 300, "")
    folders=check_tld(TESTDIR, args)
    tarred_files = tar_directories(folders,args)
    assert("00001.tar" in tarred_files)
    assert("00002.tar" in tarred_files)
    assert("00003.tar" in tarred_files)
    assert(os.path.exists(os.join(TESTDIR, "00001.tar")))
    assert(os.path.exists(os.join(TESTDIR, "00002.tar")))
    assert(os.path.exists(os.join(TESTDIR, "00003.tar")))
    os.remove(os.path.join(TESTDIR, "00001.tar"))
    os.remove(os.path.join(TESTDIR, "00002.tar"))
    os.remove(os.path.join(TESTDIR, "00003.tar"))

    #Test with compression
    args=FakeArgs(TESTDIR, False, True, 300, "")
    folders=check_tld(TESTDIR, args)
    tarred_files = tar_directories(folders,args)
    assert("00001.tar.gz" in tarred_files)
    assert("00002.tar.gz" in tarred_files)
    assert("00003.tar.gz" in tarred_files)
    assert(os.path.exists(os.join(TESTDIR, "00001.tar.gz")))
    assert(os.path.exists(os.join(TESTDIR, "00002.tar.gz")))
    assert(os.path.exists(os.join(TESTDIR, "00003.tar.gz")))
    os.remove(os.path.join(TESTDIR, "00001.tar.gz"))
    os.remove(os.path.join(TESTDIR, "00002.tar.gz"))
    os.remove(os.path.join(TESTDIR, "00003.tar.gz"))

def test_threshold_seconds():
    args=FakeArgs(TESTDIR, False, True, 300, "")
    os.mkdir(os.path.join(TESTDIR,"new_folder"))
    folders=check_tld(TESTDIR, args)
    assert("new_folder" not in folders)
    os.rmdir(os.path.join(TESTDIR,"new_folder"))

if __name__ == "__main__":
    unittest.main()