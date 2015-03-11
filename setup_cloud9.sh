#!/bin/bash

sudo apt-get install -y ipython python-matplotlib python-sklearn python-pandas htop unzip

scp ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/driven_data_countable_care/countable_care.tar.gz .
unzip -x countable_care.tar.gz
rm countable_care.tar.gz
