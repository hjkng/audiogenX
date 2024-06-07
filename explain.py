import os
import sys
# project_path = '' # absolute path of your project
# sys.path.append(project_path)
from audiocraft.models import AudioGen
from audiocraft.models import Explainer
from audiocraft.models import MaskGenerator
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from hear21passt.base import get_basic_model
import julius


def gen_mask(maskGenerators, emb):
    params=[]
    reparams=[]
    for mask_gen in maskGenerators:
        x, reparam = mask_gen(emb)
        params.append(x.squeeze())
        reparams.append(reparam.squeeze())

    params = torch.stack(params, dim=0)
    reparams = torch.stack(reparams, dim=0)
    return params, reparams

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='test', type=str)
parser.add_argument('--hard', default=False, type=bool)

parser.add_argument('--lr', default=1E-3, type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--test', default=5, type=float)

parser.add_argument('--beta', default=1E-3, type=float)
parser.add_argument('--gamma', default=1E-1, type=float)
args = parser.parse_args()

if __name__ == '__main__':
    promt_data = f"./data/{args.dataset}.csv"
    data_dir = f"./data/{args.dataset}/"
    result_dir = f'./results/{args.dataset}/'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"{result_dir} generate.")
    else:
        print(f"{result_dir} exist.")

    with open(f'{result_dir}/param.txt', "w") as param:
        param.write(str(args))

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    audio_model = AudioGen.get_pretrained('facebook/audiogen-medium')
    explainer = Explainer(audio_model, audio_model.lm)

    hear21passt_model = get_basic_model(mode="logits")
    hear21passt_model.eval()
    model = hear21passt_model.to(device)

    sequences = torch.load(f'{dir}/sequences.pt')
    conds = torch.load(f'{data_dir}/conds.pt')
    descriptions = pd.read_csv(promt_data)['caption'].tolist()

    length = sequences.shape[0]

    kl_mean =  [[] for _ in range(3)]
    mask_size = []

    epochs = args.epochs
    lr = args.lr
    test_case = args.test

    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    index = []
    for idx in tqdm(range(0, len(descriptions), 1)):
        index.append(idx)

        description = descriptions[idx]

        # load generated sequence corresponding to generated audio
        ori_sequence = sequences[idx, :, :].unsqueeze(0).to(device)
        ori_cond = conds[idx, :, :].to(device)
        audio_range = ori_cond.shape[0]

        audio_ori = explainer.token_to_audio(ori_sequence)
        audio = julius.resample_frac(audio_ori, 16000, 32000)

        # description -> t5 -> embedding
        description_emb = explainer.get_token_emb(description).squeeze().detach()

        # mask generator model initialize
        maskGenerators = nn.ModuleList(
            [MaskGenerator(description_emb.shape[0], description_emb.shape[1]).to(explainer.device) for _ in range(audio_range)])
        for mask_gen in maskGenerators:
            mask_gen.hard = args.hard

        optimizer = optim.Adam(maskGenerators.parameters(), lr=lr, weight_decay=1e-3)
        explainer.generation_params['use_sampling'] = False

        maskGenerators.train()
        EPS = 1e-6

        for epoch in range(epochs):
            params, reparams = gen_mask(maskGenerators, description_emb)

            # generate logit, emb 
            _, out_F = explainer(ori_sequence[:, :, :-1].permute(2, 1, 0), description, reparams)
            _, out_CF = explainer(ori_sequence[:, :, :-1].permute(2, 1, 0), description, 1 - reparams)

            cond_F, _ = out_F.split(out_F.shape[0]//2, dim=0)
            cond_CF, _ = out_CF.split(out_CF.shape[0]// 2, dim=0)


            loss_F = - cos(ori_cond, cond_F.squeeze()).sum()
            loss_CF = cos(ori_cond, cond_CF.squeeze()).sum()

            l1 = abs(params).sum()
            l2 = torch.sqrt((params ** 2).sum())

            loss = loss_F + loss_CF  + (l1* args.beta)  + (l2* args.gamma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch%10 == 0 or epoch==epochs-1:
                print(
                    f'epoch({epoch}) : {loss.item():.3f} | {loss_F.item():.3f} | {loss_CF.item():.3f} | {l1.item():.3f} | {l2.item():.3f}')

        explainer.generation_params['use_sampling'] = True
        explainer.generation_params['top_k'] = 250
        maskGenerators.eval()

        # Mask
        with torch.no_grad():
            params, reparams = gen_mask(maskGenerators, description_emb)
            sequence, _, _ = explainer.generate_with_mask([description] * test_case, mask=reparams)
            audio_F = explainer.token_to_audio(sequence)

        audio_F = julius.resample_frac(audio_F, 16000, 32000)
        print("factual audio done")

        # 1-Mask
        with torch.no_grad():
            sequence, _, _ = explainer.generate_with_mask([description] * test_case, mask=1-reparams)
            audio_CF = explainer.token_to_audio(sequence)

        with torch.no_grad():
            sequence, _, _ = explainer.generate_with_mask([description] * test_case)
            audio_N = explainer.token_to_audio(sequence)

        audio_N = julius.resample_frac(audio_N, 16000, 32000)
        print(f"N={test_case} audio done")

        audios = torch.cat([audio_ori, audio_F, audio_CF, audio_N]).squeeze(1)
        clipwise_output = model(audios).softmax(-1).to(device)

        kl_loss = nn.KLDivLoss(reduction='none')
        kl = []
        for i in range(1, clipwise_output.shape[0]):
            kl.append(kl_loss((clipwise_output[0]+EPS).log(), clipwise_output[i]).sum(-1).item())
        kl = torch.tensor(kl).to(device)

        kl = kl.split(test_case)
        kls = [tmp.mean().item() for tmp in kl]

        for i, mean in enumerate(kls):
            kl_mean[i].append(mean)

        mask_size.append(reparams.mean().item())


        formatted_list = [f"{a:.2f}" for a in kls]
        print(formatted_list)


        torch.save(
            {
            'params' :params.cpu(),
            'mask': reparams.cpu(),

            'audio_F' : audio_F.cpu(),
            'audio_CF' :audio_CF.cpu(),
            'audio_N' : audio_N.cpu(),
            }
        , f'{result_dir}/{idx}.pth')

        data = {
            'index' : index,

            'factual_mean' : kl_mean[0],
            'cf_mean': kl_mean[1],
            'n_mean': kl_mean[2],

            'mask': mask_size,
        }

        df = pd.DataFrame(data)
        df.to_csv(f'{result_dir}/{args.version}.csv', index=True)