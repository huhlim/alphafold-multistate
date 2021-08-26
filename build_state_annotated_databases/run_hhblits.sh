#!/bin/bash

UNICLUST30=$1

for fa in $(ls *.fa);
do
    name=$(basename $fa .fa)
    if [[ ! -e $name.a3m ]]; then
        echo "hhblits -i $fa -d $UNICLUST30 -o /dev/null -oa3m $name.a3m -n 2 -cpu 4 -v 0"
    fi
done | parallel -j 8 --workdir . :::

