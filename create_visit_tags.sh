cd /scratch/midway3/tfairnington

find . -maxdepth 1 -type d | while read -r dir; do
    d=$(basename "$dir")

    [[ "$d" =~ _V([0-9]+)(_|$) ]] || continue
    visit="V${BASH_REMATCH[1]}"

    find "$dir" -maxdepth 1 -type f | while read -r f; do
        b=$(basename "$f")

        [[ "$b" =~ _${visit}(\.[^.]+)$ ]] && continue
        [[ "$b" == *.* ]] || continue

        ext="${b##*.}"
        stem="${b%.*}"
        new="$dir/${stem}_${visit}.${ext}"

        [[ -e "$new" ]] && { echo "SKIP exists: $new"; continue; }

        mv -v "$f" "$new"
    done
done
