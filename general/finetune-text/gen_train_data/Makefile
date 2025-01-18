# Default values
INPUT_CSV ?= data/input/training_data.csv
OUTPUT_TRAIN ?= data/output/training_data.jsonl
OUTPUT_VAL ?= data/output/validation_data.jsonl
TRAIN_RATIO ?= 0.8
EMPTY_EXAMPLES ?= 2000

.PHONY: all clean process

all: run

clean:
	rm -f $(OUTPUT_TRAIN) $(OUTPUT_VAL)

run:
	python3 src/src.py \
		--input-csv $(INPUT_CSV) \
		--output-train $(OUTPUT_TRAIN) \
		--output-val $(OUTPUT_VAL) \
		--train-ratio $(TRAIN_RATIO) \
		--empty-examples $(EMPTY_EXAMPLES)

