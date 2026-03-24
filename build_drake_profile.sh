#!/bin/bash
# Download Drake's discography and build the reference profile
# Run this once — after it finishes, every fuse targets the Drake sound

PYTHON=/usr/local/Caskroom/miniconda/base/envs/vocal-fusion/bin/python
YTDLP=/usr/local/Caskroom/miniconda/base/envs/vocal-fusion/bin/yt-dlp
OUT=/Users/jpemslie/Desktop/VocalFusion/reference_tracks/drake

mkdir -p "$OUT"
cd "$OUT"

echo "=== Downloading Drake songs ==="

# Search terms to get broad coverage of Drake's catalogue
SEARCHES=(
  "Drake - God's Plan"
  "Drake - One Dance"
  "Drake - Hotline Bling"
  "Drake - Started From the Bottom"
  "Drake - HYFR"
  "Drake - Take Care"
  "Drake - Marvins Room"
  "Drake - Hold On We're Going Home"
  "Drake - Energy"
  "Drake - 0 to 100"
  "Drake - Know Yourself"
  "Drake - Forever"
  "Drake - Nice For What"
  "Drake - In My Feelings"
  "Drake - Passionfruit"
  "Drake - Gyalchester"
  "Drake - Nonstop"
  "Drake - Emotionless"
  "Drake - Scorpion"
  "Drake - Best I Ever Had"
  "Drake - Headlines"
  "Drake - Successful"
  "Drake - Thank Me Now"
  "Drake - The Motto"
  "Drake - Over"
  "Drake - Find Your Love"
  "Drake - Controlla"
  "Drake - One Dance official"
  "Drake - Too Much"
  "Drake - Jungle"
  "Drake - Legend"
  "Drake - Rich Baby Daddy"
  "Drake - Another Late Night"
  "Drake - Calling My Name"
  "Drake - Massive"
  "Drake - TSU"
  "Drake - Sticky"
  "Drake - Champagne Poetry"
  "Drake - Way 2 Sexy"
  "Drake - Girls Want Girls"
)

for QUERY in "${SEARCHES[@]}"; do
  echo "→ $QUERY"
  $YTDLP \
    --extract-audio \
    --audio-format mp3 \
    --audio-quality 0 \
    --output "%(title)s.%(ext)s" \
    --match-filter "duration > 120 & duration < 360" \
    --no-playlist \
    --ignore-errors \
    --quiet \
    "ytsearch1:$QUERY official audio"
done

echo ""
echo "=== Downloaded $(ls *.mp3 2>/dev/null | wc -l) tracks ==="
echo ""
echo "=== Building reference profile ==="
cd /Users/jpemslie/Desktop/VocalFusion
$PYTHON reference.py "$OUT"
echo ""
echo "=== Done! reference_profile.json is ready ==="
