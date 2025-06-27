#!/bin/bash
set -e

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_PLATFORM}_${BUILD_TYPE}

rm -rf ${ROOT_PWD}/build
rm -rf ${ROOT_PWD}/llm_demo
rm -rf ${ROOT_PWD}/multimodel_demo