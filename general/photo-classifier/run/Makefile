# Default values
LARGE_IMAGE_THRESHOLD ?= 3932160  # 3.7MB in bytes
TARGET_WIDTH ?= 800
TARGET_HEIGHT ?= 800
MAX_FILE_SIZE ?= 2097152  # 2MB in bytes
INPUT_DIR ?= data/input
OUTPUT_DIR ?= data/output
MODEL ?= claude-3-haiku-20240307
TEMPERATURE ?= 0.0

# Main targets
.PHONY: all clean run

all: run

clean:
	rm -rf $(OUTPUT_DIR)/*

run:
	python3 src/src.py \
		--large-image-threshold $(LARGE_IMAGE_THRESHOLD) \
		--target-width $(TARGET_WIDTH) \
		--target-height $(TARGET_HEIGHT) \
		--max-file-size $(MAX_FILE_SIZE) \
		--input-dir $(INPUT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--model $(MODEL) \
		--temperature $(TEMPERATURE)

# Example configurations
.PHONY: run-high-res run-low-res

run-high-res:
	$(MAKE) run \
		TARGET_WIDTH=1024 \
		TARGET_HEIGHT=1024 \
		MAX_FILE_SIZE=4194304

run-low-res:
	$(MAKE) run \
		TARGET_WIDTH=400 \
		TARGET_HEIGHT=400 \
		MAX_FILE_SIZE=1048576