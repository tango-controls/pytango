# ------------------------------------------------------------------------------
# This file is part of PyTango (http://pytango.rtfd.io)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

"""*build boost python script on windows*

Purpose
    Build boost-python on multiple architectures (32 and 64bits), with different toolsets (vc9, vc10),
    using different python versions.
    The different versions of boost-python DLL files are placed in a directory structure preventing
    overlapping between the different versions.
    
    PyTango Visual Studio solution configuration is compatible with the output of this script.
    
How to use it
    This script should be used together with another boost configuration file called user-config.jam.
    
    - Download boost source code from http://wwww.boost.org
    - Extract boost to a directory (ex: :file:`c:\\workspace\\boost-1.53.0`)
    - Place this file in your boost extract directory
      (ex: :file:`c:\\workspace\\boost-1.53.0\\boost_python_install.py`)
    - Place the user-config.jam file in :envvar:`%HOMEPATH%%HOMEDIR%`
    - Open a console
    - Switch to the boost directory 
    - Execute this script using python
      (ex: :file:`C:\\Python\\win32\\26\\python.exe boost_python_install.py`
"""

from __future__ import print_function


# b2 --with-python --prefix=c:\boost-1.53.0 
#    --libdir=c:\boost-1.53.0\msvc-9.0\Win32\release\shared\threading-multi\26
#    toolset=msvc-9.0 address-model=32 variant=release link=shared
#    threading=multi python=2.6 install

import os
import subprocess

boost_version = r"1.53.0"
toolsets = r"msvc-9.0", r"msvc-10.0",
address_models = ("32", "Win32"), ("64", "x64"),
variants = "release", 
links = "shared", "static",
runtime_links = ("shared", "runtime_shared"), ("static", "runtime_static"),
threadings = "multi",

pythons = "2.6", "2.7", "3.1", "3.2", "3.3",

cpus = 8
silent = True
debug_config = False
simulation = False
stage = "install"

DIV = 80*"="

# -----------------------
# overwrite defaults HERE
# -----------------------
cpus = 4

def to_name_and_dir(key):
    if isinstance(key, (str, unicode)):
        key = key, key
    return key

def main():
    try:
        _main()
    except KeyboardInterrupt:
        print("\nStopped by user")

def _main():
    global toolsets, pythons

    toolsets = r"msvc-9.0",
    pythons = "2.6", "2.7", "3.2", "3.3"   
    compile()
    
    toolsets = r"msvc-10.0",
    pythons = "3.3",    
    compile_boost()
        
def compile_boost():    
    prefix = r"c:\boost-" + boost_version
    
    silent_args = ""
    if silent:
        silent_args = "-q -d0"
    
    debug_config_args = ""
    if debug_config:
        debug_config_args = "--debug-configuration"
    
    base_cmd_line = "b2 -j{cpus} {silent} {debug_config} --with-python --build-dir={{build-dir}}".format(cpus=cpus, silent=silent_args, debug_config=debug_config_args)
    options = "prefix", "libdir", "includedir"
    properties = "toolset", "address-model", "variant", "link", "runtime-link", "threading", "python"
    
    cmd_line_template = base_cmd_line
    for option in options:
        cmd_line_template += " --{0}={{{1}}}".format(option, option)
    for prop in properties:
        cmd_line_template += " {0}={{{1}}}".format(prop, prop)
    
    cmd_line_template += " {0}".format(stage)
    fh = open("NUL", "w")
    kwargs = { "prefix" : prefix }
    for toolset in toolsets:
        kwargs["toolset"], toolset_dir = to_name_and_dir(toolset)
        for address_model in address_models:
            kwargs["address-model"], address_model_dir = to_name_and_dir(address_model)
            for variant in variants:
                kwargs["variant"], variant_dir = to_name_and_dir(variant)
                for link in links:
                    link, link_dir = to_name_and_dir(link)
                    kwargs["link"] = link
                    for runtime_link in runtime_links:
                        runtime_link, runtime_link_dir = to_name_and_dir(runtime_link)
                        kwargs["runtime-link"] = runtime_link
                        # Skip invalid compiler option
                        if link == "shared" and runtime_link == "static":
                            print("--> Skipping invalid compile option link=shared runtime_link=static <--")
                            continue
                        for threading in threadings:
                            kwargs["threading"], threading_dir = to_name_and_dir(threading)
                            for python in pythons:
                                kwargs["python"], python_dir = to_name_and_dir(python)
                                info = " ".join([ "{0}={1}".format(k, v) for k, v in kwargs.items() if k in properties ])
                                python_dir = python_dir.replace(".", "")
                                
                                lib_dir = prefix, threading_dir, variant_dir, toolset_dir, address_model_dir, link_dir, runtime_link_dir, python_dir
                                lib_dir = os.path.join(*lib_dir)
                                kwargs["libdir"] = lib_dir
                                
                                include_dir = os.path.join(prefix, "include")
                                kwargs["includedir"] = include_dir
                                
                                kwargs["build-dir"] = prefix
                                
                                cmd_line = cmd_line_template.format(**kwargs)
                                args = cmd_line.split()

                                print("Running {0}... ".format(info), end='')
                                ret = 0
                                if not simulation:
                                    p = subprocess.Popen(args, stdout=fh, stderr=fh)
                                    ret = p.wait()
                                if ret == 0:
                                    print("[OK]")
                                else:
                                    print("[FAILED]")
                                    print("\t" + cmd_line)
    fh.close()
    
if __name__ == "__main__":
    main()
