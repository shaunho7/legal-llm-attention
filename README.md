# Impacts of Continued Legal Pre-Training and IFT on LLMs' Latent Representations of Human-Defined Legal Concepts
In the Main Proceedings of the 37th International Conference on Legal Knowledge and Information Systems​ (December 11-13, 2024), Brno, Czech Republic.
Preprint link: https://arxiv.org/abs/2410.12001
This is the codebase for the experiments conducted in the above paper.

# Overview of directories and scripts
## 1_generate_raw_data
Each notebook in this folder does the following:
1. Loads a model (from the choices ['mistralai/Mistral-7B-v0.1', 'Equall/Saul-7B-Base', 'Equall/Saul-7B-Instruct-v1']) and maps it to some safe_model_id.
2. Runs a text sequence through the forward method of the model and generates the resulting attention matrix, <class 'torch.Tensor'> of shape torch.Size([32, 32, seq_len, seq_len]), where 32, 32 is num_layers, num_heads in the Mistral-7B family.
3. Saves this resulting attention matrix '{safe_sequence_id}_{safe_model_id}_attentions_raw.npy'. Each safe_sequence_id points to a given human-defined legal concept as defined in Section 2.2, Table 1 of the paper.
4. Takes the attention matrix and filters out attention to tokens of low semantic importance by resetting their values to 0. The list of tokens filtered is ["'", '(', ')', ',', '-', '.', '\:', '<s>', '▁', '▁(', '▁...', '▁[...]'].
5. Saves this resulting, filtered attention matrix as '{safe_sequence_id}_{safe_model_id}_attentions_filtered.npy'
6. Identifies the indices of the tokens representing the human-defined legal concepts. Manual selection is done in some cases here to avoid accidentally capturing duplicates tokens outside of the human-defined legal concepts such as "the", "a", etc.
7. Computes the proportion of total attention allocated to those specific token indices and represents it as a 'proportion matrix' of dimensionality num_layers, num_heads (32, 32). Each value represents the proportion of total attention allocated to indices representing human-defined legal concepts in the given attention head as identified by its layer-head index.
8. Saves this proportion matrix as '{safe_sequence_id}_{safe_model_id}_proportions.npy'.
One notebook exists for each human-defined legal concept and the text sequence that contains it, as found in the items of Table 1 in Section 2.2 of the paper. Each human-defined legal concept maps to a safe_sequence_id as indicated in the 'Mapping' section in this readme file below.

## 2_analyze_proportions_of_total_attention_S3.1_3.2
All the 'proportion matrices' generated in Step 8 above are saved to this directory. 
1. The matrices are flattened into 'proportions_data.csv' using 1_parse_data_as_DF.ipynb as found in the subdirectory statistical_analyses_S3.1_3.2.
2. 'proportions_data.csv' computes differences in the proportions of total attention to each legal concept across various model pairings, grouped by attention head (given by layer-head index) and legal concept.
3. The R scripts are used to generate the analyses found in Section 3 of the paper. Ensure that 'proportions_data.csv' is found in the same working directory when running the R scripts. Specifically:
4. 2_compute_skewness_entropy_kurtosis.R computes the skewness, kurtosis, and entropy values reported in Section 3.1, Table 2 of the paper.
5. 3_compute_layer_wise_meanplots.R plots the mean differences in attention score changes to Mistral 7B, grouped by layer depth, as reported in Section 3.2, Figure 1 of the paper.
6. Other useful plots include 4_compute_layer_wise_medianplots.R and 5_plot_QQ_plots.R which analyze the median attention score changes and tests the extent to which the attention score shifts are normally distributed.

## 3_compare_raw_attention_matrices_S3.3
All the raw attention matrices and filtered attention matrices generated in Steps 3 and 5 in the raw data generation process (see item 1 above) are saved to this directory.
The methodology of the paper is to use the filtered attention matrices. As such, only the filtered attention matrices '{safe_sequence_id}_{safe_model_id}_attentions_filtered.npy' are used in the scripts below, but they can be used to analyze the raw attention matrices with minimal modification.
NOTE: The raw and filtered attention matrices for the legal concept 'Fact Elements' is not uploaded as each of them exceeds GitHub's upload size limits (approx. 110MB each). There are large variations in the file sizes of the attention matrices between legal concepts because the lengths of the text sequences containing each concept varies and the differences in these lengths are eccentuated by the quadriatic complexity of attention computation (the size of the attention matrix is seq_len x seq_len). Kindly contact the author/owner of this repo for access to the data. 
1. 1_view_attention_matrix.py plots an attention matrix for a given model and legal concept/text sequence. This generates Figure 2 in Section 3.3 of the paper.
2. 2_view_diff_matrix.py takes two models and a legal concept/text sequence as input, and plots a matrix showing the differences in attention scores between the two models on the same text sequence. This generates Figure 3 in Section 3.3 of the paper.
 
# Mapping of experiment_ids and safe_sequence_ids to human-defined legal concepts:
Definiendum: 'facet-1',
Role: 'facet-2',      
Permissible Action: 'facet-3',
Prohibitory Action: 'facet-4',
Condition: 'facet-5'
Evidence Object: 'facet-6',
Fact Elements: 'thematic'

Kindly contact the paper's author or the owner of this repo for any questions.
