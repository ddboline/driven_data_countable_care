#!/bin/bash

scp ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/driven_data_countable_care/countable_care.tar.gz .
tar zxvf countable_care.tar.gz
rm countable_care.tar.gz

./my_model.py $1 > output_${1}.out 2> output_${1}.err

D=`date +%Y%m%d%H%M%S`
tar zcvf output_${1}_${D}.tar.gz model_${1}.pkl.gz output_${1}.out output_${1}.err
scp output_${1}_${D}.tar.gz ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/driven_data_countable_care/
ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk done_driven_data_countable_care"
