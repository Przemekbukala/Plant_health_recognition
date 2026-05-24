#!/bin/bash
DIR="$HOME/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2"

declare -A MAP=(
    ["AppleScab"]="Apple___Apple_scab" 
    ["AppleCedarRust"]="Apple___Cedar_apple_rust" 
    ["AppleBlackRot"]="Apple___Black_rot" 
    ["AppleHealthy"]="Apple___healthy"
    ["BlueberryHealthy"]="Blueberry___healthy"
    ["CherryPowderyMildew"]="Cherry_(including_sour)___Powdery_mildew" 
    ["CherryHealthy"]="Cherry_(including_sour)___healthy"
    ["CornCommonRust"]="Corn_(maize)___Common_rust_" 
    ["CornGraySpot"]="Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot" 
    ["CornNorthernBlight"]="Corn_(maize)___Northern_Leaf_Blight" 
    ["CornHealthy"]="Corn_(maize)___healthy"
    ["GrapeBlackRot"]="Grape___Black_rot" 
    ["GrapeEsca"]="Grape___Esca_(Black_Measles)" 
    ["GrapeLeafBlight"]="Grape___Leaf_blight_(Isariopsis_Leaf_Spot)" 
    ["GrapeHealthy"]="Grape___healthy"
    ["OrangeHuanglongbing"]="Orange___Haunglongbing_(Citrus_greening)"
    ["PeachBacterialSpot"]="Peach___Bacterial_spot" 
    ["PeachHealthy"]="Peach___healthy"
    ["PepperBacterialSpot"]="Pepper,_bell___Bacterial_spot" 
    ["PepperHealthy"]="Pepper,_bell___healthy"
    ["PotatoEarlyBlight"]="Potato___Early_blight" 
    ["PotatoLateBlight"]="Potato___Late_blight" 
    ["PotatoHealthy"]="Potato___healthy"
    ["RaspberryHealthy"]="Raspberry___healthy" 
    ["SoybeanHealthy"]="Soybean___healthy" 
    ["SquashPowderyMildew"]="Squash___Powdery_mildew"
    ["StrawberryLeafScorch"]="Strawberry___Leaf_scorch" 
    ["StrawberryHealthy"]="Strawberry___healthy"
    ["TomatoBacterialSpot"]="Tomato___Bacterial_spot" 
    ["TomatoEarlyBlight"]="Tomato___Early_blight" 
    ["TomatoLateBlight"]="Tomato___Late_blight" 
    ["TomatoLeafMold"]="Tomato___Leaf_Mold" 
    ["TomatoSeptoria"]="Tomato___Septoria_leaf_spot" 
    ["TomatoSpiderMites"]="Tomato___Spider_mites Two-spotted_spider_mite" 
    ["TomatoTargetSpot"]="Tomato___Target_Spot" 
    ["TomatoYellowCurlVirus"]="Tomato___Tomato_Yellow_Leaf_Curl_Virus" 
    ["TomatoMosaicVirus"]="Tomato___Tomato_mosaic_virus" 
    ["TomatoHealthy"]="Tomato___healthy"
)

mkdir -p "$DIR/test_fixed"

for FILE in "$DIR/test/test"/*.*; do
    NAME=$(basename "$FILE")
    CORE="${NAME%%[0-9]*}"
    
    if [[ -n "${MAP[$CORE]}" ]]; then
        mkdir -p "$DIR/test_fixed/${MAP[$CORE]}"
        cp "$FILE" "$DIR/test_fixed/${MAP[$CORE]}/"
    fi
done

echo "teraz test set jest w: $DIR/test_fixed"