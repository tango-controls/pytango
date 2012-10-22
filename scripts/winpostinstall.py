from __future__ import print_function

import sys
import os
import os.path

import urllib
try:
    urlretrieve = urllib.urlretrieve
except AttributeError:
    import urllib.request
    urlretrieve = urllib.urlrequest.urlretrieve

try:
    file_created
except:
    def file_created(*args, **kwargs): pass

def find_tango_root():
    program_files = os.environ["PROGRAMFILES"]
    tango = os.path.join(program_files, 'tango')
    if os.path.isdir(tango):
        return tango

def find_tango_dll():
    program_files = os.environ["PROGRAMFILES"]
    tangodll = os.path.join(program_files, 'tango','win32','lib','vc9_dll')
    if os.path.isdir(tangodll):
        return tangodll

def install_tango_dll():
    pytango_web = "http://www.tango-controls.org/static/PyTango/"
    print("Fetching appropriate tango from " + pytango_web+ "...")
    pytango_dll_path = pytango_web + "tangodll/"
    pytango_dll_file = pytango_dll_path + "tango_8.0_win32_vc9_dll.zip"
    filename, headers = urlretrieve(pytango_dll_file)
    import distutils.sysconfig
    import zipfile
    pytango_dll_zip = zipfile.ZipFile(filename)
    pytango_dir = distutils.sysconfig.get_python_lib()
    pytango_dir = os.path.join(pytango_dir, "PyTango")
    print("Extracting " + filename + " into " + pytango_dir + "...")
    pytango_dll_zip.extractall(pytango_dir)
    print("Registering all files...")
    for name in pytango_dll_zip.namelist():
        print("Registering " + name)
        name = os.path.join(pytango_dir, name)
        file_created(name)
    
def remove():
    print ("removing PyTango")
    
def install():
    tango = find_tango_dll()
    if tango is None:
        print("Could NOT find Tango C++!")
        install_tango_dll()
    else:
        print("Found tango at " + tango)
        return

def main():
    if len(sys.argv) < 2:
        op = "-install"
    else:
        op = sys.argv[1]
    
    if "-install" in op:
        install()
    elif "-remove" in op:
        remove()
    else:
        print("unknown operation " + op)
 
if __name__ == "__main__":
    try:
        main()
    except:
        import traceback
        traceback.print_exc()
        raw_input("Press any key to continue")