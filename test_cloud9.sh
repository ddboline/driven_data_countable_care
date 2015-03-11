#!/bin/bash

./my_model.py $1 > output.out 2> output.err

ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk done_driven_data_countable_care"
