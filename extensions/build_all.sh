#!/bin/bash
for file in ./*
do
    if test -d $file && test -f $file/build.sh
    then
        cd $file
        echo building $file
        bash build.sh
        if [ $? != 0 ]; then
            exit
        fi
        cd ..
    fi
done
