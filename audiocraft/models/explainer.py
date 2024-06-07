import random

import typing as tp
import numpy as np

import torch
import torch.nn as nn



from ..utils.autocast import TorchAutocast

class Explainer(nn.Module):
    def __init__(self, audiogen, lm,
                 use_sampling: bool = True, top_k: int = 250,
                 top_p: float = 0.0, temperature: float = 1.0,
                 duration: float = 5.0, cfg_coef: float = 3.0,
                 two_step_cfg: bool = False,):

        super().__init__()
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
        else:
            self.autocast = TorchAutocast(
                enabled=True, device_type=self.device.type, dtype=torch.float16)

        self.audiogen = audiogen
        self.lm = lm

        self.extend_stride = 2
        self.duration = duration
        self.generation_params = {
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'two_step_cfg': two_step_cfg,
        }

        self.lm.eval()

    def set_generation_params(self, use_sampling: bool = False, top_k: int = 250,
                              top_p: float = 0.0, temperature: float = 1.0,
                              duration: float = 1.0, cfg_coef: float = 3.0,
                              two_step_cfg: bool = False, extend_stride: float = 2):
        """Set the generation parameters for AudioGen.

        Args:
            use_sampling (bool, optional): Use sampling if True, else do argmax decoding. Defaults to True.
            top_k (int, optional): top_k used for sampling. Defaults to 250.
            top_p (float, optional): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.0.
            temperature (float, optional): Softmax temperature parameter. Defaults to 1.0.
            duration (float, optional): Duration of the generated waveform. Defaults to 10.0.
            cfg_coef (float, optional): Coefficient used for classifier free guidance. Defaults to 3.0.
            two_step_cfg (bool, optional): If True, performs 2 forward for Classifier Free Guidance,
                instead of batching together the two. This has some impact on how things
                are padded but seems to have little impact in practice.
            extend_stride: when doing extended generation (i.e. more than 10 seconds), by how much
                should we extend the audio each time. Larger values will mean less context is
                preserved, and shorter value will require extra computations.
        """
        assert extend_stride < self.max_duration, "Cannot stride by more than max generation duration."

        self.extend_stride = extend_stride
        self.duration = duration
        self.generation_params = {
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'two_step_cfg': two_step_cfg,
        }

    def split_token(self, description): # (str)
        """Tokenize the given description."""
        return self.lm.condition_provider.conditioners.description.tokenize([description])

    def get_token_size(self, description): # (str)
        """Get the size of the tokenized description."""
        tokens = self.split_token(description)
        return tokens['input_ids'].shape[-1]

    def get_token_emb(self, description): # (str)
        """Get the token embeddings for the given description. Using pretrained T5 model"""
        tokens = self.lm.condition_provider.conditioners.description.tokenize([description])
        emb, _ =  self.lm.condition_provider.conditioners.description(tokens)
        return emb

    def generate_audio(self, description): # (str)
        """Generate audio based on the provided description.
            Sample function with audiogen.generate()
        """
        gen_sequence, _, _ = self.generate_with_mask([description])

        max_gen_len = int(self.duration * self.audiogen.frame_rate)
        pattern = self.lm.pattern_provider.get_pattern(max_gen_len)
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=-1)

        audio = self.token_to_audio(out_codes)
        return audio

    def prepare_tokens_and_attributes(self, descriptions): # (str)
        """Prepare tokens and attributes for audio generation."""
        attributes, prompt_tokens = self.audiogen._prepare_tokens_and_attributes(descriptions, None)
        return attributes, prompt_tokens

    def token_to_audio(self, gen_sequence): # (torch.Tensor) (1, T, 4) : (Batch, Time, Codebook)
        """Convert generated audio token sequence to audio."""
        max_gen_len = int(self.duration * self.audiogen.frame_rate)
        pattern = self.lm.pattern_provider.get_pattern(max_gen_len)
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=-1)

        return self.audiogen.compression_model.decode(out_codes, None)

    def generate_with_mask(self,
                descriptions,

                is_reproduce: bool = False,
                gen_sequence: tp.Optional[torch.Tensor] = None,
                mask: tp.Optional[torch.Tensor] = None, ):
        """Generate audio tokens, embeddings, logits with an optional mask."""

        total_gen_len = int(self.duration * self.audiogen.frame_rate)
        attributes, prompt_tokens = self.prepare_tokens_and_attributes(descriptions)

        if is_reproduce is True:
            assert gen_sequence is not None, "gen_sequence must be provided for reproduce mode."

        with self.autocast:
            gen_sequence, outs, logits = self.lm.generate_with_mask(prompt_tokens, attributes, max_gen_len=total_gen_len,
                                                                    is_reproduce=is_reproduce, gen_sequence=gen_sequence,
                                                                    text_mask=mask, **self.generation_params)

        return gen_sequence, outs, logits

    def forward(self,
                total_sequence, # (torch.Tensor) (T, 1, 4) => (Time, Sequence, Codebook)
                descriptions,   # ([str]) prompt / description to be explained
                mask: tp.Optional[torch.Tensor] = None, # (T, D) => (Time, Description_token)
                is_training=False, # (bool) training mode
                ):
        """Give description to be explained, all audio tokens generated with the description, and mask to be trained
            return generated audio token logits and embeddings
        """
        attributes, prompt_tokens = self.prepare_tokens_and_attributes([descriptions for _ in range(total_sequence.shape[0])])
        with self.autocast:
            logit, out = self.lm.generate_sequence(total_sequence, prompt_tokens, attributes, is_training=is_training,
                                                                         text_mask=mask,
                                                                         **self.generation_params)
        return logit, out