import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from ZHMolGraph import ZHMolGraph
import pickle as pkl
import argparse

def get_seq_from_fasta(seq_file):
    """Read sequence from a FASTA file."""
    with open(seq_file, 'r') as file:
        record = next(SeqIO.parse(file, 'fasta'))  # Read the first sequence
        seq_id = record.id  # Sequence ID
        seq = str(record.seq)  # Sequence content (string)
    return seq

def show_log():
    ### print logo ###
    print('#'*64)
    print("""
     _______   ____  ___      _ _____                 _     
    |___  / | | |  \/  |     | |  __ \               | |    
       / /| |_| | .  . | ___ | | |  \/_ __ __ _ _ __ | |__  
      / / |  _  | |\/| |/ _ \| | | __| '__/ _` | '_ \| '_ \ 
    ./ /__| | | | |  | | (_) | | |_\ \ | | (_| | |_) | | | |
    \_____|_| |_|_|  |_/\___/|_|\____/_|  \__,_| .__/|_| |_|
                                               | |          
                                               |_|             
    """)
    print('#'*64, '\n')

def main():
    '''
    usage: python predict_RPI.py -r example/RNA_seq.fasta -p example/protein_seq.fasta -j test -o example/Result
    '''
    parser = argparse.ArgumentParser(description="Predict the RNA-Protein Interaction (RPI)")
    parser.add_argument("-r", "--RNA_seq_file", required=True, help="Path to the RNA sequence file (FASTA format)")
    parser.add_argument("-p", "--protein_seq_file", required=True, help="Path to the protein sequence file (FASTA format)")
    parser.add_argument("-j", "--jobname", required=True, help="Job name for this prediction task")
    parser.add_argument("-o", "--output_result_path", required=True, help="Path to save the prediction results")
    args = parser.parse_args()
    show_log()
    # Read input sequences
    RNA_seq = get_seq_from_fasta(args.RNA_seq_file)
    protein_seq = get_seq_from_fasta(args.protein_seq_file)

    print('#'*20, 'I. Input sequence of RNA and protein', '#'*20, '\n')
    print(f"Input RNA sequence: {RNA_seq} \nRNA sequence length: {len(RNA_seq)} \n")
    print(f"Input protein sequence: {protein_seq} \nprotein sequence length: {len(protein_seq)} \n")

    RNA_seq_df = pd.DataFrame([{'RNA_aa_code': RNA_seq}])
    protein_seq_df = pd.DataFrame([{'target_aa_code': protein_seq}])


    print('#'*10, 'II. Load LLMs model and LLMs embedding from model dataset.', '#'*10, '\n')

    # Load pre-trained embeddings
    model_Dataset = 'NPInter2'
    with open('data/Mol2Vec/RPI_' + model_Dataset + '_rnafm_embed_normal.pkl', 'rb') as file:
        rnas = pkl.load(file)

    with open('data/Mol2Vec/RPI_' + model_Dataset + '_proteinprottrans_embed_normal.pkl', 'rb') as file:
        proteins = pkl.load(file)

    # Create ZHMolGraph object
    vecnn_object = ZHMolGraph.ZHMolGraph(
        interactions_location=f'data/interactions/dataset_RPI_{model_Dataset}_RP.csv',
        interactions=None,
        interaction_y_name='Y',
        rnas_dataframe=rnas,
        rna_seq_name='RNA_aa_code',
        proteins_dataframe=proteins,
        protein_seq_name='target_aa_code',
        model_out_dir=f'trained_model/ZHMolGraph_VecNN_model_RPI_{model_Dataset}/',
        debug=False
    )

    # Normalize embeddings
    rnas_embeddings_array = np.array(rnas['normalized_embeddings'].tolist())
    vecnn_object.mean_rna_embeddings = np.mean(rnas_embeddings_array, axis=0)
    vecnn_object.centered_rna_embeddings = rnas_embeddings_array - vecnn_object.mean_rna_embeddings
    vecnn_object.centered_rna_embeddings_length = np.mean(
        np.sqrt(np.sum(vecnn_object.centered_rna_embeddings * vecnn_object.centered_rna_embeddings, axis=1))
    )

    proteins_embeddings_array = np.array(proteins['normalized_embeddings'].tolist())
    vecnn_object.mean_protein_embeddings = np.mean(proteins_embeddings_array, axis=0)
    vecnn_object.centered_protein_embeddings = proteins_embeddings_array - vecnn_object.mean_protein_embeddings
    vecnn_object.centered_protein_embeddings_length = np.mean(
        np.sqrt(np.sum(vecnn_object.centered_protein_embeddings * vecnn_object.centered_protein_embeddings, axis=1))
    )

    # Generate embeddings for input sequences
    test_rna = vecnn_object.get_rnafm_embeddings(prediction_interactions=RNA_seq_df,
                                                 replace_dataframe=False,
                                                 return_normalisation_conststants=True)
    test_protein = vecnn_object.get_ProtTrans_embeddings(prediction_interactions=protein_seq_df,
                                                         replace_dataframe=False,
                                                         return_normalisation_conststants=True)

    vecnn_object.rnas_dataframe = test_rna
    vecnn_object.rna_list = list(vecnn_object.rnas_dataframe[vecnn_object.rna_seq_name])
    vecnn_object.proteins_dataframe = test_protein
    vecnn_object.protein_list = list(vecnn_object.proteins_dataframe[vecnn_object.protein_seq_name])

    # Predict binding probability
    interactions_seqpairs = pd.concat([RNA_seq_df, protein_seq_df], axis=1)
    rna_vector_length = 640
    protein_vector_length = 1024


    print('\n', '#'*20, 'III. Load trained model and testing.', '#'*20, '\n')


    vecnn_object.predict_RPI(model_dataset=model_Dataset,
                             graphsage_path=vecnn_object.model_out_dir,
                             jobname=args.jobname,
                             test_dataframe=interactions_seqpairs,
                             rna_vector_length=rna_vector_length,
                             protein_vector_length=protein_vector_length,
                             rnas=test_rna,
                             proteins=test_protein,
                             embedding_type='Pretrain',
                             graphsage_embedding=1)

    print('\n', '#'*20, 'IV. Output binding probability.', '#'*20, '\n')

    # Save results
    os.makedirs(args.output_result_path, exist_ok=True)
    result_file = os.path.join(args.output_result_path, f'{args.jobname}.txt')
    scores = vecnn_object.averaged_results
    average_score = sum(scores) / len(scores)

    with open(result_file, "w") as file:
        file.write(f"RNA seq: {RNA_seq}\n")
        file.write(f"Protein seq: {protein_seq}\n")
        file.write(f"Probability score: {average_score:.3f}\n")

if __name__ == "__main__":
    main()
