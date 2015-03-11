#!/bin/bash

./my_model.py $1 > output.out 2> output.err

ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk done_driven_data_countable_care"
tar zcvf output_${1}_`date +%Y%m%d%H%M%S`.tar.gz model_*.pkl.gz output.out output.err
scp output_*.tar.gz ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/driven_data_countable_care/
