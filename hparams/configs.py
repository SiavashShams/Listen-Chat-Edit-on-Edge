from utils.task import *
from data.datasets.special_prompts import *

configs = {
    '2Speech2Audio': {
        'path': '/engram/naplab/shared/TextSpeechVGGMix/manifests/TextrolSpeech2VGGSound2Mix/prompts',
        'task_handler': TaskHandler2Speech2Audio,
        'special_prompts': special_prompts_2speech_2audio

    },
    '2Speech2FSD': {
        'path': '/engram/naplab/shared/TextSpeechVGGMix/manifests/TextrolSpeech2FSD50k2Mix/prompts',
        'task_handler': TaskHandler2Speech2Audio,
        'special_prompts': special_prompts_2speech_2audio

    },
    '2Speech2FSD_unseen': {
        'path': '/engram/naplab/shared/TextSpeechVGGMix/manifests/TextrolSpeech2FSD50k2Mix_unseen/prompts',
        'task_handler': TaskHandler2Speech2Audio,
        'special_prompts': special_prompts_2speech_2audio

    },
    '3Speech1Audio': {
        'path': '/engram/naplab/shared/TextSpeechVGGMix/manifests/TextrolSpeech3VGGSound1Mix/prompts',
        'task_handler': TaskHandler3Speech1Audio,
        'special_prompts': special_prompts_3speech_1audio
    },
    '1Speech3Audio': {
        'path': '/engram/naplab/shared/TextSpeechVGGMix/manifests/TextrolSpeech1VGGSound3Mix/prompts',
        'task_handler': TaskHandler1Speech3Audio,
        'special_prompts': special_prompts_1speech_3audio    
    },
    '2Speech1Audio': {
        'path': '/engram/naplab/shared/TextSpeechVGGMix/manifests/TextrolSpeech2VGGSound1Mix/prompts',
        'task_handler': TaskHandler2Speech1Audio,
        'special_prompts': special_prompts_2speech_1audio    
    },
    '1Speech2Audio': {
        'path': '/engram/naplab/shared/TextSpeechVGGMix/manifests/TextrolSpeech1VGGSound2Mix/prompts',
        'task_handler': TaskHandler1Speech2Audio,
        'special_prompts': special_prompts_1speech_2audio
    },
    '2Speech': {
        'path': '/engram/naplab/shared/TextSpeechVGGMix/manifests/TextrolSpeech2Mix/prompts',
        'task_handler': TaskHandler2Speech,
        'special_prompts': special_prompts_2speech
    },
    '2Audio': {
        'path': '/engram/naplab/shared/TextSpeechVGGMix/manifests/VGGSound2Mix/prompts',
        'task_handler': TaskHandler2Audio,
        'special_prompts': special_prompts_2audio
    },
    '2SpeechTSE': {
        'path': '/engram/naplab/shared/TextSpeechVGGMix/manifests/TextrolSpeech2Mix_TSE/prompts',
        'task_handler': TaskHandler2Speech,
        'special_prompts': special_prompts_2speech
    },
}

def get_task_handler(case):
    return configs[case]['task_handler']

def get_manifest_path(case):
    return configs[case]['path']

def get_special_prompts(case):
    return configs[case]['special_prompts']
