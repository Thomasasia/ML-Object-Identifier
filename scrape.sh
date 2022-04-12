#!/bin/sh

for i in 1 2
do
  echo "cat images"
  python3 bbid.py -s "cat" --limit 1000 -o ./images/cats
  python3 bbid.py -s "cat image" --limit 1000 -o ./images/cats
  python3 bbid.py -s "cat picture" --limit 1000 -o ./images/cats

  echo "dog images"
  python3 bbid.py -s "dog" --limit 1000 -o ./images/dogs
  python3 bbid.py -s "dog image" --limit 1000 -o ./images/dogs
  python3 bbid.py -s "dog picture" --limit 1000 -o ./images/dogs

  echo "goat images"
  python3 bbid.py -s "goat" --limit 1000 -o ./images/goats
  python3 bbid.py -s "goat image" --limit 1000 -o ./images/goats
  python3 bbid.py -s "goat picture" --limit 1000 -o ./images/goats

  echo "Generic images"
  python3 bbid.py -s "image" --limit 1000 -o ./images/msc
  python3 bbid.py -s "scenery" --limit 500 -o ./images/msc
  python3 bbid.py -s "pub" --limit 200 -o ./images/msc
  python3 bbid.py -s "pokemon" --limit 200 -o ./images/msc
  
done
