#!/bin/bash

# Cache All Variations Script
# Builds caches for all combinations of cluster, input_length, and max_nodata parameters
# This ensures fast loading for all future training runs

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
CLUSTERS=("1" "3")
INPUT_LENGTHS=(12 6 3)
MAX_NODATA_VALUES=(0.5 0.75)

# Calculate total combinations
TOTAL_COMBINATIONS=$((${#CLUSTERS[@]} * ${#INPUT_LENGTHS[@]} * ${#MAX_NODATA_VALUES[@]}))
echo "📊 Total combinations to process: $TOTAL_COMBINATIONS"
echo ""

# Counter for progress tracking
CURRENT=0

# Loop through all combinations
for cluster in "${CLUSTERS[@]}"; do
    for input_length in "${INPUT_LENGTHS[@]}"; do
        for max_nodata in "${MAX_NODATA_VALUES[@]}"; do
            CURRENT=$((CURRENT + 1))
            
            echo "=" >> /dev/stderr
            echo "🔧 Processing combination $CURRENT/$TOTAL_COMBINATIONS" >> /dev/stderr
            echo "   Cluster: $cluster" >> /dev/stderr
            echo "   Input Length: $input_length" >> /dev/stderr
            echo "   Max NoData: $max_nodata" >> /dev/stderr
            echo "=" >> /dev/stderr
            
            # Run setup_data.py with current parameter combination
            python setup_data.py \
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
                echo "✅ Success: cluster=$cluster, input_length=$input_length, max_nodata=$max_nodata" >> /dev/stderr
            else
                echo "❌ Failed: cluster=$cluster, input_length=$input_length, max_nodata=$max_nodata" >> /dev/stderr
                echo "Continuing with next combination..." >> /dev/stderr
            fi
            
            echo "" >> /dev/stderr
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
