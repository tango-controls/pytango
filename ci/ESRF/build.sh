#!/bin/bash

INSTALL_DIR=/segfs/tango/ci/PyTango

if [ ! -z "$NODE_NAME" -a -f "$NODE_NAME" ]
then
	if [ $NODE_NAME = "ct32windows7" ]
	then
        ./$NODE_NAME
    else
        source "$NODE_NAME"
    fi
else
    echo "The settings file for the node $NODE_NAME does not exist!"
    echo "Create ci/$INSTITUTE/$NODE_NAME file."
fi

cd ../..
#used system by default

rm -rf build/*

case "${realos}" in
	"debian6_64")
        /usr/bin/python setup.py build
		;; 
	"redhate"*)
        #redhate4/5
        /segfs/bliss/bin/python2.6 setup.py build
        ;;
    "windows7_32")
        #VC9
        /cygdrive/c/Python27_32/python setup.py build
        ;;
esac

if [ $? != 0 ]
then
	exit $?
fi

case "${realos}" in
	"debian6_64")
        rm -rf $INSTALL_DIR/debian6/*
		/usr/bin/python setup.py install --prefix=$INSTALL_DIR/debian6
		;; 
	"redhate4_32")
        rm -rf $INSTALL_DIR/redhate4/*
		/segfs/bliss/bin/python2.6 setup.py install --prefix=$INSTALL_DIR/redhate4
		;;
	"redhate5_64")
        rm -rf $INSTALL_DIR/redhate5/*
		/segfs/bliss/bin/python2.6 setup.py install --prefix=$INSTALL_DIR/redhate5
		;;
    "windows7_32")
        /cygdrive/c/Python27_32/python setup.py install --prefix=$INSTALL_DIR/w7_32_VC9
        ;;
	*)
		echo "Not supporting operating system: " ${OSTYPE}
        exit $?
		;;
esac


exit $?


