Downloading Cord-19
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2021-02-15/document_parses.tar.gz

For preparing LM data
run: python prep_pdf.py
run: python create_lm_data.py