#!/usr/bin/env bash
rm dist/nrgten*.tar.gz
sh create_package.sh
pip install dist/nrgten*.tar.gz --no-index
