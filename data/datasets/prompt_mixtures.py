import os
import json
import math
import random
from itertools import product

import numpy as np
import librosa
import soundfile as sf

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset

from data.datasets.prompt_templates import powerset, \
    build_short_style_descriptions_for_two_speakers, \
    build_style_description

SR = 16000

def fix_length(wav, n_sample=4*16000, rand_crop=False):
    assert wav.ndim == 1
    if len(wav) > n_sample:
        if rand_crop:
            rand_start = np.random.randint(0,  len(wav)-n_sample)
            wav = wav[rand_start:rand_start+n_sample]
        else:
            wav = wav[:n_sample]
    elif len(wav) < n_sample:
        wav = np.pad(wav, (0, n_sample-len(wav)), mode='constant', constant_values=0)
    return wav


class PromptMixtures(Dataset):

    def __init__(
        self, 
        manifest_files,
        select_n=0,
        duration='default',
        task_handler=None,
        rand_tasks=False, # depreciated if prob_gpt_prompt==1
        acts=['0', '1', 'D' ,'U'],
        tasks=['HE', 'HVC', 'OVC', 'RHVC', 
        'SE', 'SR', 'S↑', 'S↓', 'TAE', 'TAR', 
        'TA↑', 'TA↓', 'TSE', 'TSR', 'TS↑', 'TS↓'],
        volume_scale=2, # scale the wave magnitude by 2 (6dB)
        prob_gpt_prompt=1, # the probability to use GPT-generated prompt, 1 means always
        rand_prompt=False,
        prompt_builder=None,
        delta_styles=False,
        special_prompts={},
        prob_special_prompt=0,
        ret_tar=False,
        ret_src=False,
        ret_dict=True
    ):
        super().__init__()
        self.manifest_files = manifest_files
        print(f'Fetched {len(manifest_files)} manifest files.')

        self.records = []
        for manifest_file in manifest_files:
            with open(manifest_file, 'r') as json_file:
                self.records += json.load(json_file)

        if select_n > 0:
            self.records = self.records[:select_n]

        self.duration = duration
        self.rand_tasks = rand_tasks
        self.acts = acts
        self.acts.sort()
        self.tasks = tasks
        self.tasks.sort()
        self.task_handler = task_handler
        self.volume_scale = volume_scale
        self.prob_gpt_prompt = prob_gpt_prompt
        self.rand_prompt = rand_prompt
        self.prompt_builder = prompt_builder
        self.delta_styles = delta_styles
        self.special_prompts = special_prompts
        self.prob_special_prompt = prob_special_prompt
        self.ret_tar = ret_tar
        self.ret_src = ret_src
        self.ret_dict = ret_dict
        assert prob_gpt_prompt < 1 or (prob_gpt_prompt == 1 and not rand_tasks)
        
        print('Actions supported: ', acts, ' with volume_scale =', volume_scale)
        print('Tasks supported: ', tasks)
        print(f'Use GPT prompts with prob {str(prob_gpt_prompt)} and handcrafted prompts with prob {str(1-prob_gpt_prompt)}.')

    def __len__(self):
        return len(self.records)
        
    def edit_source(self, src, act):
        if act == '0':
            tar = src * 0
        elif act == '1':
            tar = src * 1
        elif act == 'U':
            tar = src * self.volume_scale
        elif act == 'D':
            tar = src / self.volume_scale
        else:
            raise ValueError(str(act))
        return tar

    def __getitem__(self, idx, acts=None):

        record = self.records[idx]

        if acts != None:
            rand_tasks = False
            # use_gpt_prompt if generated for acts
            use_gpt_prompt = (acts==record['acts'])
            acts = acts
            task = self.task_handler.group_task(acts)
        else:
            rand_tasks = self.rand_tasks
            use_gpt_prompt = (random.uniform(0, 1)<=self.prob_gpt_prompt)
            if use_gpt_prompt:
                task = record['task']
                acts = record['acts']
            else:
                task = random.choice(self.tasks) if rand_tasks else record['task']
                acts = random.choice(self.task_handler.ungroup_task[task]) \
                    if rand_tasks else record['acts']
        
        # Use special instructions for extraction, removal or volume control problems
        # if targets are either all speeches or all non-speech sounds
        if self.prob_special_prompt > 0 and acts in self.special_prompts \
        and (\
            # Randomly use special prompts in training
            (self.rand_prompt and random.uniform(0, 1)<=self.prob_special_prompt) or \
            # Always assign self.prob_special_prompt fraction of testing examples 
            # with special prompts. idx//73 acts as hashing,
            ((not self.rand_prompt) and ((idx//37)%int(1/self.prob_special_prompt)==0))\
        ):
            use_special_prompts = True
        else: 
            use_special_prompts = False

        if use_special_prompts:
            special_task_prompts = self.special_prompts[acts]
            if self.rand_prompt:
                prompt = random.choice(special_task_prompts)
            else:
                prompt = special_task_prompts[(idx//41)%len(special_task_prompts)]

        # Use GPT generated prompts
        elif use_gpt_prompt:
            gpt_prompts = record['parsed_prompts']
            if self.rand_prompt: # randomly choose one from prompts
                prompt = random.choice(gpt_prompts)
            else: # use the last generated prompt
                prompt = gpt_prompts[-1] # the last one

        # Use handcrafted prompts
        else:
            if self.delta_styles:
                style1, style2 = build_short_style_descriptions_for_two_speakers(
                    record['gender1'], record['pitch1'], record['tempo1'], record['energy1'], record['emotion1'],
                    record['gender2'], record['pitch2'], record['tempo2'], record['energy2'], record['emotion2']
                )
            else:
                style1 = build_style_description(
                    gender=record['gender1'],
                    pitch=record['pitch1'],
                    tempo=record['tempo1'],
                    energy=record['energy1'],
                    emotion=record['emotion1']
                )
                style2 = build_style_description(
                    gender=record['gender2'],
                    pitch=record['pitch2'],
                    tempo=record['tempo2'],
                    energy=record['energy2'],
                    emotion=record['emotion2']
                )

            spks = [style1, style2]
            labels = [record['label3'], record['label4']]
            prompt = self.prompt_builder(acts, spks=spks, labels=labels) 

        n_src = 4 if 'path4' in record else \
            3 if 'path3' in record else \
            2 if 'path2' in record else \
            ValueError(-1)

        mix_path = record['path']
        src_paths = [record[f'path{i}'] for i in range(1, n_src+1)]

        # Load input mixture
        mix, sr = sf.read(mix_path, dtype='float32')

        # Load every source
        srcs = [sf.read(src_path, dtype='float32')[0] for src_path in src_paths]

        # Edit every source as the target
        tars = [self.edit_source(src, act) for src, act in zip(srcs, acts)]
        tar = sum(tars)
            
        if self.duration == 'default':
            duration = float(record['duration'])
        else:
            duration = float(self.duration)

        samples = int(duration*sr)
        tar = fix_length(tar, samples, rand_crop=False) # (N)
        mix = fix_length(mix, samples, rand_crop=False) # (N)

        out = (mix, tar, prompt, acts)
        
        if self.ret_src: # return original sources
            out += (np.stack(srcs, axis=-1), )

        if self.ret_tar: # return edited sources
            out += (np.stack(tars, axis=-1), )
        if self.ret_dict:
            style1 = build_style_description(
                gender=record['gender1'],
                pitch=record['pitch1'],
                tempo=record['tempo1'],
                energy=record['energy1'],
                emotion=record['emotion1']
            )
            style2 = build_style_description(
                gender=record['gender2'],
                pitch=record['pitch2'],
                tempo=record['tempo2'],
                energy=record['energy2'],
                emotion=record['emotion2']
            )
            record['style1'] = style1
            record['style2'] = style2

            out += (record, )

        return out
