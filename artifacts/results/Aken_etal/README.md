# Aken et al. 2022

Title: Assertion Detection in Clinical Notes: Medical Language Models to the Rescue?
GitHub: https://github.com/bvanaken/clinical-assertion-data
HuggingFace: https://huggingface.co/bvanaken/clinical-assertion-negation-bert
## Tables

1. Table 1: Distribution of text types and classes in the three employed datasets. Note that possible is a minority class across datasets as well as text types. In the i2b2 dataset, for instance, only 5% of all labels are possible. 
2. Table 2: Results of baseline approaches and (medical) language models on the i2b2 Assertions Task. Pre-trained medical language models outperform all earlier approaches – with a large margin on the possible class. Note that Bhatia et al. (2019) only evaluated their model on negation detection.
3. Table 3: Experimental results (in F1) for the best per-forming Bio+Discharge Summary BERT model on two further assertion datasets and their different text types. Both datasets were not seen during training. Note that the number of evaluation samples is very low for some text types (i.e. possible class in nurse letters), which impairs the expressiveness of these results.
