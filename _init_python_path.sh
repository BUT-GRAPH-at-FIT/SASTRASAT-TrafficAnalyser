#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)"
echo "Setting PYTHONPATH=$DIR"
export PYTHONPATH=$DIR
