#!/bin/bash

progress=0
files=($directory/clips/*)
total=${#files[@]}
echo "Total files: $total"
for f in ${files[@]}; do
#for ((i=274755; i<$total; i++)); do
    base=$(basename "${files[$i]}")
    ffmpeg -n -hide_banner -loglevel error -i "${files[$i]}" -ac 1 -ar 16000 "$directory/clips_wav/${base%.*}.wav"
    echo "Progress: $i/$total"
done
