#!/bin/sh

for i in 1 2
do
  python3 bbid.py -s "cat" --limit 1000 -o ./images/cats
  python3 bbid.py -s "cat image" --limit 1000 -o ./images/cats
  python3 bbid.py -s "cat picture" --limit 1000 -o ./images/cats
done
