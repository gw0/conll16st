# Run baseline discourse sense classifier with most common class

# conll16st - English datasets
mkdir ../ex/majority-en ../ex/majority-en/{trial,valid,test,blind}

DATA_DIR='../data/conll16st-en-03-29-16-trial'
OUTPUT_DIR='../ex/majority-en/trial'
./sample_sup_parser.py en "$DATA_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR"
./tira_sup_eval.py "$DATA_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/tira_sup_eval.log"

DATA_DIR='../data/conll16st-en-03-29-16-dev'
OUTPUT_DIR='../ex/majority-en/valid'
./sample_sup_parser.py en "$DATA_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR"
./tira_sup_eval.py "$DATA_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/tira_sup_eval.log"

DATA_DIR='../data/conll16st-en-03-29-16-test'
OUTPUT_DIR='../ex/majority-en/test'
./sample_sup_parser.py en "$DATA_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR"
./tira_sup_eval.py "$DATA_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/tira_sup_eval.log"

DATA_DIR='../data/conll15st-en-03-29-16-blind-test'
OUTPUT_DIR='../ex/majority-en/blind'
./sample_sup_parser.py en "$DATA_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR"
./tira_sup_eval.py "$DATA_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/tira_sup_eval.log"

# conll16st - Chinese datasets
mkdir ../ex/majority-zh ../ex/majority-zh/{valid,test,blind}

DATA_DIR='../data/conll16st-zh-01-08-2016-dev'
OUTPUT_DIR='../ex/majority-zh/valid'
./sample_sup_parser.py zh "$DATA_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR"
./tira_sup_eval.py "$DATA_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/tira_sup_eval.log"

DATA_DIR='../data/conll16st-zh-01-08-2016-test'
OUTPUT_DIR='../ex/majority-zh/test'
./sample_sup_parser.py zh "$DATA_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR"
./tira_sup_eval.py "$DATA_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/tira_sup_eval.log"

DATA_DIR='../data/conll16st-zh-04-27-2016-blind-test'
OUTPUT_DIR='../ex/majority-zh/blind'
./sample_sup_parser.py zh "$DATA_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR"
./tira_sup_eval.py "$DATA_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/tira_sup_eval.log"
