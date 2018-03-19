# Run baseline discourse sense classifier with most common class

mkdir ../ex/majority-en ../ex/majority-en/{trial,train,valid,test,blind}
./sample_sup_parser.py en ../data/conll16st-en-03-29-16-trial ../ex/majority-en/trial/ ../ex/majority-en/trial/
./sample_sup_parser.py en ../data/conll16st-en-03-29-16-train ../ex/majority-en/train/ ../ex/majority-en/train/
./sample_sup_parser.py en ../data/conll16st-en-03-29-16-dev ../ex/majority-en/valid/ ../ex/majority-en/valid/
./sample_sup_parser.py en ../data/conll16st-en-03-29-16-test ../ex/majority-en/test/ ../ex/majority-en/test/
./sample_sup_parser.py en ../data/conll15st-en-03-29-16-blind-test ../ex/majority-en/blind/ ../ex/majority-en/blind/

mkdir ../ex/majority-zh ../ex/majority-zh/{train,valid,test,blind}
./sample_sup_parser.py zh ../data/conll16st-zh-01-08-2016-train ../ex/majority-en/train/ ../ex/majority-en/train/
./sample_sup_parser.py zh ../data/conll16st-zh-01-08-2016-dev ../ex/majority-en/valid/ ../ex/majority-en/valid/
./sample_sup_parser.py zh ../data/conll16st-zh-01-08-2016-test ../ex/majority-en/test/ ../ex/majority-en/test/
./sample_sup_parser.py zh ../data/conll16st-zh-04-27-2016-blind-test ../ex/majority-en/blind/ ../ex/majority-en/blind/
