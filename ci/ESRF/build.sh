#!/bin/bash
#export CYGWIN="${CYGWIN} nodosfilewarning"
INSTALL_DIR=/segfs/tango/ci/PyTango
export PATH=$PATH:/cygdrive/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio\ 9.0/Common7/IDE 

if [ ! -z "$NODE_NAME" -a -f "$NODE_NAME" ]
then
	echo $NODE_NAME
	if [ $NODE_NAME != "ct32windows7" ]
	then
        source "$NODE_NAME"
    fi
else
    echo "The settings file for the node $NODE_NAME does not exist!"
    echo "Create ci/$INSTITUTE/$NODE_NAME file."
fi

cd ../..
#used system by default

rm -rf build/*

print $realos
case "${realos}" in
	"debian"*)
		#debian6/7
        /usr/bin/python setup.py build
		;; 
	"redhate"*)
        #redhate4/5
        /segfs/bliss/bin/python2.6 setup.py build
        ;;
	"windows7")
		pyVersion="py26, py27"
		platform="Win32, x64"
		SLN="C:\jenkins\workspace\PyTango\OperatingSystems\Windows64-VC10\win\PyTango_VS9\PyTango.sln"
		LIBNAME="PyTango.vcproj"
		#devenv="C:\Program Files (x86)\Microsoft Visual Studio 9.0\Common7\IDE\devenv.exe" 
		cd /cygdrive/c
		/bin/rm -rf /cygdrive/c/Temp/pytango
		pyVersion="py26"
		platform="Win32"
		SLN="C:\jenkins\workspace\PyTango\OperatingSystems\Windows64-VC10\win\PyTango_VS9\PyTango.sln"
		LIBNAME="PyTango.vcproj"
		#devenv="\cygdrive\c\Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio\ 9.0/Common7/IDE/devenv.exe"
		for pv in $pyVersion
		do
		   for p in $platform
		   do
			   pyN=${pv:2:4}
			   MODE=$pv"_bopystatic_tangostatic_release|"$p
			   OUTFILE="c:/Temp/log_${p}_"${pyN}
			   MAKE_CMD="devenv.exe $SLN /project PyTango.vcproj /rebuild $MODE /projectconfig $MODE /out $OUTFILE"
			   $MAKE_CMD
		   done
		done
        ;;
	*)
		echo "Build - Not supporting operating system: " ${OSTYPE}
        exit $?
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
	"debian7_64")
        rm -rf $INSTALL_DIR/debian7/*
		/usr/bin/python setup.py install --prefix=$INSTALL_DIR/debian7
		;; 
	"redhate4_32")
        rm -rf $INSTALL_DIR/redhate4/*
		/segfs/bliss/bin/python2.6 setup.py install --prefix=$INSTALL_DIR/redhate4
		;;
	"redhate5_64")
        rm -rf $INSTALL_DIR/redhate5/*
		/segfs/bliss/bin/python2.6 setup.py install --prefix=$INSTALL_DIR/redhate5
		;;
    "windows7"*)
		INSTALL_DIR="//unixhome/segfs/tango/ci/PyTango/windows7"
		/bin/rm -rf $INSTALL_DIR/* || echo "Error executing rm command"
        /bin/cp -rf /cygdrive/c/Temp/pytango/build_8.1.0_tg8.1.2_boost1.53.0/lib  $INSTALL_DIR || echo "Error executing cp command, LIB"
		/bin/cp -rf /cygdrive/c/Temp/pytango/build_8.1.0_tg8.1.2_boost1.53.0/dist  $INSTALL_DIR || echo "Error executing cp command, DIST"
		/bin/chmod -R 755  $INSTALL_DIR || echo "Error executing chmod command"
        ;;
	*)
		echo "Install - Not supporting operating system: " ${OSTYPE}
        exit $?
		;;
esac


exit $?




