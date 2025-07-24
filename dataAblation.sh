#!/bin/bash


echo "🚀 Building caches for all parameter combinations..."
echo "This will pre-build caches for fast training startup."
echo ""

# Base parameters (consistent across all runs)
DATASET_ROOT="./Data/ML"
OUTPUT_LENGTH=1
TRAIN_YEARS="2013 2014 2015 2016 2017 2018 2019 2020 2021"
VAL_YEARS="2022 2023"
TEST_YEARS="2024 2025"
REMOVED_CHANNELS=("'DEM' 'red' 'green' 'blue' 'ndvi' 'ndwi' 'ndbi' 'albedo'", "LST")
CLUSTERS=("1" "all" "2" "3" "4")
INPUT_LENGTHS=(12)
MAX_NODATA_VALUES=(0.5)

# Calculate total combinations
TOTAL_COMBINATIONS=$((${#CLUSTERS[@]} * ${#INPUT_LENGTHS[@]} * ${#MAX_NODATA_VALUES[@]} * ${#LEARNING_RATES[@]} * ${#PRECISIONS[@]} * ${#REMOVED_CHANNELS[@]}))
log_and_echo "📊 Total data runs to execute: $TOTAL_COMBINATIONS"
log_and_echo ""

# Counter for progress tracking
CURRENT=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

# Loop through all combinations
for cluster in "${CLUSTERS[@]}"; do
    for input_length in "${INPUT_LENGTHS[@]}"; do
        for max_nodata in "${MAX_NODATA_VALUES[@]}"; do
            for removed_channels in "${REMOVED_CHANNELS[@]}"; do
                CURRENT=$((CURRENT + 1))

                echo "="
                echo "🎯 Starting training run $CURRENT/$TOTAL_COMBINATIONS"
                echo "   Cluster: $cluster"
                echo "   Input Length: $input_length"
                echo "   Max NoData: $max_nodata"
                echo "   Learning Rate: $learning_rate"
                echo "   Precision: $precision"
                echo "   Model: $MODEL_SIZE"
                echo "   Removed ChannelsL $removed_channels"
                echo "="

                # Run training with current parameter combination
                python setup_data.py \
                    --removed_channels $removed_channels
                    --dataset_root "$DATASET_ROOT" \
                    --cluster "$cluster" \
                    --input_length $input_length \
                    --output_length $OUTPUT_LENGTH \
                    --train_years $TRAIN_YEARS \
                    --val_years $VAL_YEARS \
                    --test_years $TEST_YEARS \
                    --max_nodata $max_nodata \

                    # Check if the command succeeded
                if [ $? -eq 0 ]; then
                    echo "✅ Success: cluster=$cluster, input_length=$input_length, max_nodata=$max_nodata" >> /dev/stderr
                else
                    echo "❌ Failed: cluster=$cluster, input_length=$input_length, max_nodata=$max_nodata" >> /dev/stderr
                    echo "Continuing with next combination..." >> /dev/stderr
                fi

                echo "" >> /dev/stderr
            done
        done
    done
done

echo "🎉 Cache building complete!" >> /dev/stderr
echo "" >> /dev/stderr
echo "📈 Summary of cached configurations:" >> /dev/stderr
echo "   Clusters: ${CLUSTERS[*]}" >> /dev/stderr
echo "   Input Lengths: ${INPUT_LENGTHS[*]}" >> /dev/stderr
echo "   Max NoData Values: ${MAX_NODATA_VALUES[*]}" >> /dev/stderr
echo "   Total: $TOTAL_COMBINATIONS combinations" >> /dev/stderr
echo "" >> /dev/stderr
echo "🚀 You can now run training with any of these parameter combinations for instant startup!" >> /dev/stderr
echo "" >> /dev/stderr
echo "Example training commands:" >> /dev/stderr
echo "   python train_with_cache.py --cluster 1 --input_length 12 --max_nodata 0.5" >> /dev/stderr
echo "   python train_with_cache.py --cluster 3 --input_length 6 --max_nodata 0.75" >> /dev/stderr