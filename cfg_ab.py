from math import fabs
from turtle import pd

import torch

AminoAcid_Vocab = {
    "<pad>": 0,
    "A":1,
    "R":2,
    "N":3, 
    "D":4,
    "C":5,
    "Q":6,
    "E":7,
    "G":8,
    "H":9,
    "I":10,
    "L":11,
    "K":12,
    "M":13,
    "F":14,
    "P":15,
    "S":16,
    "T":17,
    "W":18,
    "Y":19,
    "V":20,
    #"<unk>": 22,  
    "X":21,  # <END>

}
#AA_LIST= ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','<unk>','X']
AA_LIST = [
    "<pad>",  # index 0
    "A", "R", "N", "D", "C", "Q", "E", "G", "H",
    "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",  # indices 1-20
    "X"   # index 21
]
PHYSICAL_PROPERTIES = {
    '<pad>':[0.0,0.0,0.0,0.0],
    'A': [ 1.8,  0.0,  88.6, 1.0], # 疏水性, 电荷, 体积, 可及性
    'R': [-4.5,  1.0, 173.4, 0.5],
    'N': [-3.5,  0.0, 132.1, 0.8],
    'D': [-3.5, -1.0, 133.1, 0.8],
    'C': [ 2.5,  0.0, 121.0, 0.6],
    'Q': [-3.5,  0.0, 146.1, 0.7],
    'E': [-3.5, -1.0, 147.1, 0.7],
    'G': [-0.4,  0.0,  75.1, 1.0],
    'H': [-3.2,  0.5, 155.1, 0.6],
    'I': [ 4.5,  0.0, 166.7, 0.4],
    'L': [ 3.8,  0.0, 166.7, 0.4],
    'K': [-3.9,  1.0, 174.2, 0.5],
    'M': [ 1.9,  0.0, 162.9, 0.5],
    'F': [ 2.8,  0.0, 189.9, 0.3],
    'P': [-1.6,  0.0, 155.1, 0.7],
    'S': [-0.8,  0.0, 105.1, 0.9],
    'T': [-0.7,  0.0, 119.1, 0.8],
    'W': [-0.9,  0.0, 227.8, 0.2],
    'Y': [-1.3,  0.0, 193.6, 0.3],
    'V': [ 4.2,  0.0, 140.0, 0.5],
    #'<unk>':[0.0, 0.0, 0.0, 0.0],
    'X': [ 0.0,  0.0, 100.0, 0.5],
}

prop_lookup = torch.tensor([PHYSICAL_PROPERTIES[aa] for aa in AA_LIST], dtype=torch.float32)

def seq_to_prop(seq):
    """
    seq: [B, L] (LongTensor)
    prop_lookup: [vocab_size, prop_dim]
    return: [B, L, prop_dim]
    """
    return prop_lookup[seq]  # 自动索引

class configuration():
    def __init__(self,
                    hidden_size: int = 1024,
                    max_position_embeddings: int = 256,
                    type_residue_size: int = 9,
                    layer_norm_eps: float = 1e-12,
                    hidden_dropout_prob = 0.1,
                    use_bias = True,
                    initializer_range=0.02,
                    num_hidden_layers = 4,
                    type_embedding=False,
                    ) -> None:
        
        self.AminoAcid_Vocab = AminoAcid_Vocab
        self.token_size = len(self.AminoAcid_Vocab)
        self.residue_size = 22
        self.hidden_size = hidden_size
        self.pad_token_id = 0
        self.max_position_embeddings = max_position_embeddings
        self.type_residue_size = type_residue_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use__bias = use_bias
        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range
        self.type_embedding = type_embedding