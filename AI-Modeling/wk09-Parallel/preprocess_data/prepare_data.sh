
input_file="industry_corpus_0_10000.jsonl"


# Call the Python script with the list of file paths in the chunk
python preprocess_data.py \
  --input $input_file \
  --tokenizer-type "GPT2BPETokenizer" \
  --vocab-file "gpt2/vocab.json" \
  --merge-file "gpt2/merges.txt" \
  --output-prefix "output_data" \

