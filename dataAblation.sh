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

# Parameter variations
CLUSTERS=("1" "all" "2" "3" "4")
INPUT_LENGTHS=(12)
MAX_NODATA_VALUES=(0.5)

# Channel ablation configurations
REMOVED_CHANNELS_CONFIGS=(
    "DEM red green blue ndvi ndwi ndbi albedo"
    "LST"
)

# Calculate total combinations
TOTAL_COMBINATIONS=$((${#CLUSTERS[@]} * ${#INPUT_LENGTHS[@]} * ${#MAX_NODATA_VALUES[@]} * ${#REMOVED_CHANNELS_CONFIGS[@]}))
echo "📊 Total data runs to execute: $TOTAL_COMBINATIONS"
echo ""

# Counter for progress tracking
CURRENT=0

# Loop through all combinations
for cluster in "${CLUSTERS[@]}"; do
    for input_length in "${INPUT_LENGTHS[@]}"; do
        for max_nodata in "${MAX_NODATA_VALUES[@]}"; do
            for removed_channels in "${REMOVED_CHANNELS_CONFIGS[@]}"; do
                CURRENT=$((CURRENT + 1))

                echo "================================================="
                echo "🎯 Starting data setup run $CURRENT/$TOTAL_COMBINATIONS"
                echo "   Cluster: $cluster"
                echo "   Input Length: $input_length"
                echo "   Max NoData: $max_nodata"
                echo "   Removed Channels: $removed_channels"
                echo "================================================="

                # Run data setup with current parameter combination
                python setup_data.py \
                    --remove_channels $removed_channels \
                    --dataset_root "$DATASET_ROOT" \
                    --cluster "$cluster" \
                    --input_length $input_length \
                    --output_length $OUTPUT_LENGTH \
                    --train_years $TRAIN_YEARS \
                    --val_years $VAL_YEARS \
                    --test_years $TEST_YEARS \
                    --max_nodata $max_nodata

                # Check if the command succeeded
                if [ $? -eq 0 ]; then
                    echo "✅ Success: cluster=$cluster, input_length=$input_length, max_nodata=$max_nodata, removed_channels=[$removed_channels]" >&2
                else
                    echo "❌ Failed: cluster=$cluster, input_length=$input_length, max_nodata=$max_nodata, removed_channels=[$removed_channels]" >&2
                    echo "Continuing with next combination..." >&2
                fi

                echo "" >&2
            done
        done
    done
done

echo "🎉 Cache building complete!" >&2
echo "" >&2
echo "📈 Summary of cached configurations:" >&2
echo "   Clusters: ${CLUSTERS[*]}" >&2
echo "   Input Lengths: ${INPUT_LENGTHS[*]}" >&2
echo "   Max NoData Values: ${MAX_NODATA_VALUES[*]}" >&2
echo "   Channel Configurations: ${#REMOVED_CHANNELS_CONFIGS[@]} configs" >&2
echo "   Total: $TOTAL_COMBINATIONS combinations" >&2
echo "" >&2
echo "🚀 You can now run training with any of these parameter combinations for instant startup!" >&2