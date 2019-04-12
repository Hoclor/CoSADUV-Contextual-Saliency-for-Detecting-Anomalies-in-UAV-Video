# Converts all .avi files in train, val, test, train/targets, val/targets, and test/targets into libx264 codec

# Keyframe interval should be similar to the desired sequence lengths (1 for image handling, 150 for video handling
KEYFRAME_INTERVAL=1


for folder in "train" "val" "test"
do
  cd "$folder"
  shopt -s nullglob
  for file in *.avi
  do
    echo "$folder/$file"
    ffmpeg -hide_banner -loglevel panic -i "$file" -map v:0 -c:v libx264 -crf 18 -pix_fmt yuv420p -g $KEYFRAME_INTERVAL -profile:v high "new_$file"
    rm "$file"
    mv "new_$file" "$file"
  done
  shopt -u nullglob
  cd "targets"
  shopt -s nullglob
  for file in *.avi
  do
    echo "$folder/targets/$file"
    ffmpeg -hide_banner -loglevel panic -i "$file" -map v:0 -c:v libx264 -crf 18 -pix_fmt yuv420p -g $KEYFRAME_INTERVAL -profile:v high "new_$file"
    rm "$file"
    mv "new_$file" "$file"
  done
  shopt -u nullglob
  cd "../.."
done
echo "done"    
