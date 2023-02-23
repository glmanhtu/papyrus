#!/bin/bash

PYTHON='python'
SCRIPT='train.py'
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -gpu|--gpu-id)
    GPU_ID="$2"
    shift # past argument
    shift # past value
    ;;
    -p|--python)
    PYTHON="$2"
    shift # past argument
    shift # past value
    ;;
    -sc|--script)
    SCRIPT="$2"
    shift # past argument
    shift # past value
    ;;
    -s|--start)
    START="$2"
    shift # past argument
    shift # past value
    ;;
    -n|--name)
    NAME="$2"
    shift # past argument
    shift # past value
    ;;
    -g|--group)
    GROUP="$2"
    shift # past argument
    shift # past value
    ;;
    -e|--end)
    END="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

export CUDA_VISIBLE_DEVICES="$GPU_ID"

for i in $(seq $START $END)
do
   ${PYTHON} -u ${SCRIPT} --group "${GROUP}" --name "${NAME}_${i}" "${POSITIONAL[@]}" &> ${NAME}_${i}.log
done