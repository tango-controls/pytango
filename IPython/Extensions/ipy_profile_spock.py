""" IPython 'spock' profile, to preload PyTango and offer a friendly interface to Tango."""

import IPython.ipapi
import ipy_defaults

def main():
    ip = IPython.ipapi.get()
    try:
        ip.ex("import IPython.ipapi")
        ip.ex("import PyTango.ipython")
        ip.ex("PyTango.ipython.init_ipython(IPython.ipapi.get())")
    except ImportError:
        print "Unable to start spock profile, is PyTango installed?"

main()