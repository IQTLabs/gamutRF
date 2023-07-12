import time
import os
import pytest
import random
import shutil
import string
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

class DummyFileStructure:
    def __init__(
        self,
        dir=None,
    ):
        self.dir=dir
        self.subdirs=list()
        self.files=list()

@pytest.fixture(scope="function")
def ensure_testdir():
    letters = string.ascii_lowercase
    test_dir = os.path.join(TESTDIR, ''.join(random.choice(letters) for i in range(10))) 
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    yield test_dir

    if os.path.exists(test_dir):
        print(f"removing test directory: {test_dir}")
        shutil.rmtree(test_dir)

@pytest.fixture
def populate_dirs(ensure_testdir):
    files=list()
    letters = string.ascii_lowercase
    tfs = DummyFileStructure(ensure_testdir)
    for i in range(1,4):
        d=os.path.join(tfs.dir, f'{i:05}')
        os.mkdir(d)
        tfs.subdirs.append(d)

        fname = ''.join(random.choice(letters) for i in range(10))
        fpath = os.path.join(d,f'{fname}.txt')
        with open(fpath, 'w') as f:
            f.write(''.join(random.choice(letters) for i in range(50)))
        tfs.files.append(fpath)

    threshold_s=3
    time.sleep(threshold_s)
    yield tfs

    for f in tfs.files:
        print(f"removing file: {f}")
        os.remove(f)
    for d in tfs.subdirs:
        print(f"removing directory: {d}")
        os.rmdir(d)

def test_argument_parser():
    argument_parser()

def test_check_valid_tld(populate_dirs):
    args=FakeArgs(populate_dirs.dir, False, True, 3, "")
    folders=check_tld(populate_dirs.dir, args)
    assert(os.path.join(populate_dirs.dir,"00001") in folders)
    assert(os.path.join(populate_dirs.dir,"00002") in folders)
    assert(os.path.join(populate_dirs.dir,"00003") in folders)
    assert(os.path.join(populate_dirs.dir,"scan.csv") not in folders)

def test_check_invalid_tld():
    INVALID_TESTDIR=TESTDIR+"_invalid"
    args=FakeArgs(INVALID_TESTDIR, False, True, 3, "")
    folders=check_tld(TESTDIR, args)
    assert(folders.__sizeof__ == 0)

def test_tar_directories(populate_dirs):
    #Test without compression
    args=FakeArgs(populate_dirs.dir, False, False, 3, "")
    folders=check_tld(populate_dirs.dir, args)
    tarred_files = tar_directories(folders,args)
    assert("00001.tar" in tarred_files)
    assert("00002.tar" in tarred_files)
    assert("00003.tar" in tarred_files)
    assert(os.path.exists(os.join(populate_dirs.dir, "00001.tar")))
    assert(os.path.exists(os.join(populate_dirs.dir, "00002.tar")))
    assert(os.path.exists(os.join(populate_dirs.dir, "00003.tar")))

    #Test with compression
    args=FakeArgs(populate_dirs.dir, False, True, 3, "")
    folders=check_tld(populate_dirs.dir, args)
    tarred_files = tar_directories(folders,args)
    assert("00001.tar.gz" in tarred_files)
    assert("00002.tar.gz" in tarred_files)
    assert("00003.tar.gz" in tarred_files)
    assert(os.path.exists(os.join(populate_dirs.dir, "00001.tar.gz")))
    assert(os.path.exists(os.join(populate_dirs.dir, "00002.tar.gz")))
    assert(os.path.exists(os.join(populate_dirs.dir, "00003.tar.gz")))


def test_threshold_seconds():
    args=FakeArgs(TESTDIR, False, True, 3, "")
    os.mkdir(os.path.join(TESTDIR,"new_folder"))
    folders=check_tld(TESTDIR, args)
    assert("new_folder" not in folders)
    os.rmdir(os.path.join(TESTDIR,"new_folder"))

if __name__ == "__main__":
    unittest.main()
