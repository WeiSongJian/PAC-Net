import os
import pandas as pd
from abnumber import Chain
from joblib import Parallel, delayed
from tqdm import tqdm

def process_sequence(seq_index, sequence, scheme):
    output = {'Seq_Index': seq_index}
    try:
        ab_chain = Chain(sequence, scheme=scheme)
        if ab_chain.chain_type == 'H':
            output.update({
                'Seq_Index': seq_index,
                'Framework_1': ab_chain.fr1_seq or 'None',
                'Complementarity_1': ab_chain.cdr1_seq or 'None',
                'Framework_2': ab_chain.fr2_seq or 'None',
                'Complementarity_2': ab_chain.cdr2_seq or 'None',
                'Framework_3': ab_chain.fr3_seq or 'None',
                'Complementarity_3': ab_chain.cdr3_seq or 'None',
                'Framework_4': ab_chain.fr4_seq or 'None'
            })
            return output
    except Exception as error:
        output.update({
            'Framework_1': '', 'Complementarity_1': '',
            'Framework_2': '', 'Complementarity_2': '',
            'Framework_3': '', 'Complementarity_3': '',
            'Framework_4': ''
        })
    
    return output

def process_file(input_filepath, output_filepath, scheme_type):
    dataframe = pd.read_csv(input_filepath)
    parallel_processor = Parallel(n_jobs=-1, backend="loky")
    processed_list = parallel_processor(
        delayed(process_sequence)(idx, sequence, scheme_type) 
        for idx, sequence in tqdm(dataframe['vh'].dropna().iteritems(), desc="Processing Antibody Sequences")
    )
    processed_dataframe = pd.DataFrame(processed_list).set_index('Seq_Index')

    for column in processed_dataframe.columns:
        dataframe[column] = processed_dataframe[column]

    dataframe.to_csv(output_filepath, index=False)

def run_processing(input_path, output_path, scheme):
    process_file(input_path, output_path, scheme)

if __name__ == "__main__":
    working_directory = "/PACNET/"
    input_file_path = ""
    output_file_name = input_file_path.split('.')[2] if '.' in input_file_path else "output"
    output_file_path = ""

    os.chdir(working_directory)

    primary_scheme = 'chothia'
    backup_scheme = 'Heavy_fv_oas_train_filtered'

    run_processing(input_file_path, output_file_path, primary_scheme)
    print("Processing Completed!")
