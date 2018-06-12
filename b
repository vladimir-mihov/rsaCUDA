#! /bin/bash

CC='g++'
NVCC='nvcc'
CCOPTS='-c --std=c++11'
LINKOPTS='-lboost_program_options'

if [[ $1 == '' ]] ; then
echo 'Compiling ...'
find . -name '*.cpp' -execdir $CC $CCOPTS {} \;
$NVCC $CCOPTS main.cu
echo 'Linking ...'
find . -name '*.o' -exec $NVCC $LINKOPTS -o main {} +
fi

if [[ $1 == 'clean' ]] ; then
echo 'Cleaning object and executable files ...'
find . -name '*.o' -exec rm -f {} +
rm -f main zad15.png
fi

if [[ $@ =~ 'parser' ]] ; then
echo 'Compiling parser ...'
$CC $CCOPTS -o parser/parser.o parser/parser.cpp
fi

if [[ $@ =~ 'main' ]] ; then
echo 'Compiling main ...'
$NVCC $CCOPTS main.cu
fi

if [[ $@ =~ 'link' ]] ; then
echo 'Linking ...'
find . -name '*.o' -exec $NVCC $LINKOPTS -o main {} +
fi
