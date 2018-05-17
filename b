#! /bin/bash

CC='g++'
CCOPTS='-c --std=c++11'
LINKOPTS='--std=c++11 -lboost_program_options'

if [[ $1 == '' ]] ; then
echo 'Compiling...'
find . -name '*.cpp' -execdir $CC $CCOPTS {} \;
nvcc $CCOPTS main.cu
echo 'Linking...'
find . -name '*.o' -exec nvcc $LINKOPTS -o main {} +
fi

if [[ $1 == 'clean' ]] ; then
find . -name '*.o' -exec rm -f {} +
rm -f main
fi

if [[ $@ =~ 'parser' ]] ; then
echo 'Compiling parser'
$CC $CCOPTS -o parser/parser.o parser/parser.cpp
fi

if [[ $@ =~ 'main' ]] ; then
echo 'Compiling main'
nvcc $CCOPTS main.cu
fi

if [[ $@ =~ 'link' ]] ; then
find . -name '*.o' -exec nvcc $LINKOPTS -o main {} +
fi
