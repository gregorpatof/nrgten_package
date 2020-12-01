#!/usr/bin/env bash
rm dits/nrgten*.tar.gz
sh create_package.sh
pip install dist/nrgten*.tar.gz --no-index
