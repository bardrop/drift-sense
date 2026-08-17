#!/bin/sh
# Downloads the demo's web images (Wikimedia Commons) into demo/web_images/.
set -e
cd "$(dirname "$0")/web_images" 2>/dev/null || { mkdir -p "$(dirname "$0")/web_images"; cd "$(dirname "$0")/web_images"; }
UA="Mozilla/5.0 (drift-sense-demo)"
curl -sL -A "$UA" -o sem_kidney_stone.jpg "https://upload.wikimedia.org/wikipedia/commons/5/5d/Crystals_of_Weddellite_on_the_surface_of_a_kidney_stone_%28SEM%2C_30_KV%2C_no.14%29.jpg"
curl -sL -A "$UA" -o sem_virions.jpg "https://upload.wikimedia.org/wikipedia/commons/2/21/Influenza_A_virions_on_cilia_human_nasal_epithelium_SEM_high_magnification.jpg"
curl -sL -A "$UA" -o chip_die_ad580.jpg "https://upload.wikimedia.org/wikipedia/commons/0/0a/AD580.jpg"
curl -sL -A "$UA" -o chip_die_amd.jpg "https://upload.wikimedia.org/wikipedia/commons/f/fc/AMD3101E.jpg"
curl -sL -A "$UA" -o silicon_rings.jpg "https://upload.wikimedia.org/wikipedia/commons/f/f0/Silicon_Rings_%2810747924673%29.jpg"
echo "done"
