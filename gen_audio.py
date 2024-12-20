import os
import sys
# project_path = '' # absolute path of your project
# sys.path.append(project_path)
from audiocraft.models import AudioGen
from audiocraft.models import Explainer
import torch
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='test', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    audiogen = AudioGen.get_pretrained('facebook/audiogen-medium')
    explainer = Explainer(audiogen, audiogen.lm)

    dir = f"./data/{args.dataset}"
    promt_data = f"./{dir}/{args.dataset}.csv"

    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"{dir} generate.")
    else:
        print(f"{dir} exist.")

    explainer.duration = 5
    explainer.generation_params['use_sampling'] = True

    df = pd.read_csv(promt_data)

    length = len(df['caption'])

    batch = 10

    sequences = []
    conds = []
    logits = []
    for idx in tqdm(range(0, length, batch)):
        description = df['caption'][idx: idx+batch].tolist()
        with torch.no_grad():
            sequence, _, outs = explainer.generate_with_mask(description)
            sequence = sequence.detach().cpu()
            outs = outs.detach().cpu()

        B = outs.shape[0]
        cond, _ = outs.split(B//2, dim=0)

        sequences.append(sequence)
        conds.append(cond)

    sequences = torch.cat(sequences, dim=0)
    conds = torch.cat(conds, dim=0)

    print(sequences.shape, conds.shape)

    torch.save(sequences, f'{dir}/sequences.pt')
    torch.save(conds, f'{dir}/conds.pt')
