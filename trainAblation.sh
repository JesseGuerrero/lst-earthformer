#!/bin/bash

# Training All Variations Script
# Runs training for all combinations of cluster, input_length, max_nodata, learning_rate, and precision parameters
# Uses pre-built caches from setup_data.py for fast startup

LOG_FILE="train_ablation.txt"

# Function to log and display messages
log_and_echo() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Function to log error messages
log_error() {
    echo "$1" | tee -a "$LOG_FILE" >&2
}

log_and_echo "🚀 Running training for all parameter combinations..."
log_and_echo "This assumes caches have been pre-built with dataAblation.sh"
log_and_echo ""

# Base parameters (consistent across all runs)
DATASET_ROOT="./Data/ML"
OUTPUT_LENGTH=1
TRAIN_YEARS="2013 2014 2015 2016 2017 2018 2019 2020 2021"
VAL_YEARS="2022 2023"
TEST_YEARS="2024 2025"

# Training-specific parameters
WANDB_PROJECT="AAAI-Project-channel-ablation"
LEARNING_RATES=(0.0005)
BATCH_SIZE=32
MAX_EPOCHS=200
NUM_WORKERS=32
DEVICE=0
GPUS=2
PRECISIONS=(32)
MODEL_SIZE="earthnet"
LIMIT_TRAIN_BATCHES=1.0
LIMIT_VAL_BATCHES=1.0
LIMIT_TEST_BATCHES=1.0

# Parameter variations (matching cache building script)
CLUSTERS=("1" "all" "2" "3" "4")
INPUT_LENGTHS=(12)
MAX_NODATA_VALUES=(0.5)

# Channel ablation configurations (must match dataAblation.sh exactly)
REMOVED_CHANNELS_CONFIGS=(
    "DEM red green blue ndvi ndwi ndbi albedo"
    "LST"
)

# Calculate total combinations
TOTAL_COMBINATIONS=$((${#CLUSTERS[@]} * ${#INPUT_LENGTHS[@]} * ${#MAX_NODATA_VALUES[@]} * ${#LEARNING_RATES[@]} * ${#PRECISIONS[@]} * ${#REMOVED_CHANNELS_CONFIGS[@]}))
log_and_echo "📊 Total training runs to execute: $TOTAL_COMBINATIONS"
log_and_echo ""

# Counter for progress tracking
CURRENT=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

# Loop through all combinations
for cluster in "${CLUSTERS[@]}"; do
    for input_length in "${INPUT_LENGTHS[@]}"; do
        for max_nodata in "${MAX_NODATA_VALUES[@]}"; do
            for learning_rate in "${LEARNING_RATES[@]}"; do
                for precision in "${PRECISIONS[@]}"; do
                    for removed_channels in "${REMOVED_CHANNELS_CONFIGS[@]}"; do
                        CURRENT=$((CURRENT + 1))

                        log_error "================================================="
                        log_error "🎯 Starting training run $CURRENT/$TOTAL_COMBINATIONS"
                        log_error "   Cluster: $cluster"
                        log_error "   Input Length: $input_length"
                        log_error "   Max NoData: $max_nodata"
                        log_error "   Learning Rate: $learning_rate"
                        log_error "   Precision: $precision"
                        log_error "   Model: $MODEL_SIZE"
                        log_error "   Removed Channels: $removed_channels"
                        log_error "================================================="

                        # Run training with current parameter combination
                        python train_with_cache.py \
                            --remove_channels $removed_channels \
                            --dataset_root "$DATASET_ROOT" \
                            --cluster "$cluster" \
                            --input_length $input_length \
                            --output_length $OUTPUT_LENGTH \
                            --train_years $TRAIN_YEARS \
                            --val_years $VAL_YEARS \
                            --test_years $TEST_YEARS \
                            --max_nodata $max_nodata \
                            --wandb_project "$WANDB_PROJECT" \
                            --learning_rate $learning_rate \
                            --batch_size $BATCH_SIZE \
                            --max_epochs $MAX_EPOCHS \
                            --num_workers $NUM_WORKERS \
                            --gpus $GPUS \
                            --device $DEVICE \
                            --precision $precision \
                            --model_size "$MODEL_SIZE" \
                            --limit_train_batches $LIMIT_TRAIN_BATCHES \
                            --limit_val_batches $LIMIT_VAL_BATCHES \
                            --limit_test_batches $LIMIT_TEST_BATCHES

                        # Check if the training succeeded
                        if [ $? -eq 0 ]; then
                            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
                            log_and_echo "✅ Training Success: cluster=$cluster, input_length=$input_length, max_nodata=$max_nodata, lr=$learning_rate, precision=$precision, removed_channels=[$removed_channels]"
                        else
                            FAILED_RUNS=$((FAILED_RUNS + 1))
                            log_error "❌ Training Failed: cluster=$cluster, input_length=$input_length, max_nodata=$max_nodata, lr=$learning_rate, precision=$precision, removed_channels=[$removed_channels]"
                            log_error "Continuing with next combination..."
                        fi

                        log_and_echo ""

                        # Optional: Add a short delay between runs to prevent resource conflicts
                        # sleep 10
                    done
                done
            done
        done
    done
done

log_and_echo "🎉 All training runs completed!"
log_and_echo ""
log_and_echo "📈 Training Summary:"
log_and_echo "   Total Runs: $TOTAL_COMBINATIONS"
log_and_echo "   Successful: $SUCCESSFUL_RUNS"
log_and_echo "   Failed: $FAILED_RUNS"
log_and_echo ""
log_and_echo "📊 Trained Configurations:"
log_and_echo "   Clusters: ${CLUSTERS[*]}"
log_and_echo "   Input Lengths: ${INPUT_LENGTHS[*]}"
log_and_echo "   Max NoData Values: ${MAX_NODATA_VALUES[*]}"
log_and_echo "   Learning Rates: ${LEARNING_RATES[*]}"
log_and_echo "   Precisions: ${PRECISIONS[*]}"
log_and_echo "   Channel Configs: ${#REMOVED_CHANNELS_CONFIGS[@]} configurations"
log_and_echo "   Model Size: $MODEL_SIZE"
log_and_echo "   Batch Size: $BATCH_SIZE"
log_and_echo "   Max Epochs: $MAX_EPOCHS"
log_and_echo ""
log_and_echo "🔗 Check your WandB project '$WANDB_PROJECT' for all experiment results!"