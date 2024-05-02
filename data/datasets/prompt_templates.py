import random
from itertools import chain, combinations


def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))


def build_style_description_from(style, keys):
    description = 'the'
    if 'emotion' in keys:
        if style['emotion'] != 'neutral':
            emotion = style['emotion']
        else:
            emotion = 'neutral-emotion'
        description += ' '
        description += emotion
    if 'gender' in keys:
        description += ' '
        description += 'female' if style['gender'] == 'F' else 'male'

    description += ' speaker'

    if 'pitch' in keys or 'tempo' in keys or 'energy' in keys:
        description += ' characterized by'
    else:
        return description

    keys = [key for key in ['pitch', 'tempo', 'energy'] if key in keys]

    for i, key in enumerate(keys):
        if i == 0:
            description += ' '
        elif i == len(keys) - 1:
            if len(keys) > 2:
                description += ', and '
            else:
                description += ' and '
        else:
            description += ', '
            
        description += style[key]
        description += ' '
        description += key

    return description


def build_short_style_descriptions_for_two_speakers(
    gender1, pitch1, tempo1, energy1, emotion1,
    gender2, pitch2, tempo2, energy2, emotion2
):
    '''
    Describe speakers only by the style difference.
    Let (x1, y1, z1) and (x2, y2, z2) be the style differece tuples
    if they differ in three of five attributes. For example, x is gender, 
    y is energy, z is emotion. We then choose nonempty subsets from tuples
    e.g. (x1, z1) and (z2). Finally, we generate prompts from (x1, z1) and (z2). 
    The goal is generate more human-like prompt, because we may not describe the 
    speakers in all aspects, but only in some aspects in which speakers differ.
    '''

    s1 = (gender1, pitch1, tempo1, energy1, emotion1)
    s2 = (gender2, pitch2, tempo2, energy2, emotion2)

    # Two speakers must be distinguishable in the mixture
    assert s1 != s2

    keys = ['gender', 'pitch', 'tempo', 'energy', 'emotion']
    keys_d = [] # all attributes they differ
    s1_d = {} # all attributes and values spk1 differs from spk2
    s2_d = {} # all attributes and values spk2 differs from spk1
    for key, v1, v2 in zip(keys, s1, s2):
        if v1 != v2:
            s1_d[key] = v1
            s2_d[key] = v2
            keys_d.append(key)

    # Select a subset of all attributes they differ 
    keys_d = random.choice(powerset(keys_d))

    description1 = build_style_description_from(s1_d, keys_d)
    description2 = build_style_description_from(s2_d, keys_d)

    return description1, description2


def build_short_style_descriptions_for_one_speaker(
    gender1, pitch1, tempo1, energy1, emotion1,
):
    keys = ['gender', 'pitch', 'tempo', 'energy', 'emotion']
    s1 = {'gender': gender1, 'pitch': pitch1, 'tempo': tempo1, 
        'energy': energy1, 'emotion': emotion1}

    # Select a subset of all attributes
    keys = random.choice(powerset(keys))

    description1 = build_style_description_from(s1, keys)

    return description1


def build_short_style_descriptions_for_three_speakers(
    gender1, pitch1, tempo1, energy1, emotion1,
    gender2, pitch2, tempo2, energy2, emotion2,
    gender3, pitch3, tempo3, energy3, emotion3
):
    s1 = (gender1, pitch1, tempo1, energy1, emotion1)
    s2 = (gender2, pitch2, tempo2, energy2, emotion2)
    s3 = (gender3, pitch3, tempo3, energy3, emotion3)

    # Two speakers must be distinguishable in the mixture
    assert s1 != s2 and s1 != s3 and s2 != s3

    keys = ['gender', 'pitch', 'tempo', 'energy', 'emotion']
    keys_d = [] # all attributes they differ
    s1_d = {}
    s2_d = {}
    s3_d = {} 
    for key, v1, v2, v3 in zip(keys, s1, s2, s3):
        if v1 != v2 or v1 != v3 or v2 != v3: # or: symmetric difference
            s1_d[key] = v1
            s2_d[key] = v2
            s3_d[key] = v3
            keys_d.append(key)

    while True:
        # Select a subset of all attributes they differ 
        keys_dd = random.choice(powerset(keys_d))

        description1 = build_style_description_from(s1_d, keys_dd)
        description2 = build_style_description_from(s2_d, keys_dd)
        description3 = build_style_description_from(s3_d, keys_dd)

        if description1 != description2 and \
           description1 != description3 and \
           description2 != description3:
           break

    return description1, description2, description3


def add_synonym_keywords(hand_prompt, labels):

    shorten_labels = {
        'subway, metro, underground': 'metro',
        'police car (siren)': 'police siren',
        'bee, wasp, etc. buzzing': 'buzzing',
        'vehicle horn, car horn, honking': 'car horn',
        'rowboat, canoe, kayak rowing': 'rowboat'
    }

    labels_ = []
    for label in labels:
        if label in shorten_labels:
            labels_.append(shorten_labels[label])
        else:
            labels_.append(label)
        
    words = []
    for word in ['male', 'female', 'pitch', 'tempo', 'energy', 'volume'] + labels_:
        if word in hand_prompt:
            words.append(word)

    if len(words) == 0:
        sentence = ''
    elif len(words) == 1:
        sentence = words[0]
    elif len(words) == 2:
        sentence = words[0] + ' and ' + words[1]
    else:
        sentence = ', '.join(words[:-1]) + ' and ' + words[-1]

    return sentence


class EasyTemplateFor4Speakers():

    def __init__(
        self,
        acts=['0', '1', 'D' ,'U'],
        shuffle=True,
        random=False
    ):
        acts.sort()
        assert acts == ['0', '1', 'D' ,'U']        
        self.lookups = {
            '0': ['remove {}', 'eliminate {}', 'take {} away'],
            '1': ['keep {} as the original', 'maintain {}', 'preserve {}'],
            'D': ['turn down the volume of {}', 'lower the volume of {}', 'make {} quieter'],
            'U': ['turn up the volume of {}', 'raise the volume of {}', 'make {} louder']
        }
        self.templates = [
            'Please {}, {}, {}, and {}.',
            'I want to {}, {}, {}, and {}.',
            'Can you {}, {}, {}, and {}?'
        ]
        self.shuffle = shuffle
        self.random = random

        print(f'Initialized {str(self.__class__.__name__)}: ')
        print(f'shuffle: {str(shuffle)} random: {str(random)}')

    def __call__(self, acts, labels=None, spks=None):
        label3, label4 = labels
        src1, src2 = spks
        src3 = 'the {} sound'.format(label3)
        src4 = 'the {} sound'.format(label4)

        act1, act2, act3, act4 = acts

        if self.random:
            verb1 = random.choice(self.lookups[act1])
            verb2 = random.choice(self.lookups[act2])
            verb3 = random.choice(self.lookups[act3])
            verb4 = random.choice(self.lookups[act4])
            template = random.choice(self.templates)
        else:
            verb1 = self.lookups[act1][0]
            verb2 = self.lookups[act2][0]
            verb3 = self.lookups[act3][0]
            verb4 = self.lookups[act4][0]
            template = self.templates[0]

        phase1 = verb1.format(src1)
        phase2 = verb2.format(src2)
        phase3 = verb3.format(src3)
        phase4 = verb4.format(src4)

        if self.shuffle:
            phases = [phase1, phase2, phase3, phase4]
            random.shuffle(phases)
            phase1, phase2, phase3, phase4 = phases

        prompt = template.format(phase1, phase2, phase3, phase4)
             
        return prompt


class ShortTemplate():

    def __init__(
        self,
        acts=['0', '1', 'D' ,'U'],
        shuffle=True,
        random=False,
    ):
        acts.sort()
        assert acts == ['0', '1', 'D' ,'U']        
        self.lookups = {
            '0': ['remove {}', 'eliminate {}', 'take {} away'],
            # '1': ['extract {}', 'only keep {}', 'isolate {}'],
            'D': ['turn down the volume of {}', 'lower the volume of {}', 'make {} quieter'],
            'U': ['turn up the volume of {}', 'raise the volume of {}', 'make {} louder']
        }

        self.lookups_extract_or_remove = {
            '0': ['remove', 'eliminate', 'take away {}'],
            '1': ['extract', 'only keep', 'isolate'],
        }

        self.templates = {
            1: [
                'Please {}.',
                'I want to {}.',
                'Can you {}?'
            ],
            2: [
                'Please {} and {}.',
                'I want to {} and {}.',
                'Can you {} and {}?'
            ],
            3: [
                'Please {}, {}, and {}.',
                'I want to {}, {}, and {}.',
                'Can you {}, {}, and {}?'
            ],
            4: [
                'Please {}, {}, {}, and {}.',
                'I want to {}, {}, {}, and {}.',
                'Can you {}, {}, {}, and {}?'
            ]
        }

        self.templates_extract_or_remove = {
            1: [
                'Please {} {}.',
                'I want to {} {}.',
                'Can you {} {}?'
            ],
            2: [
                'Please {} {} and {}.',
                'I want to {} {} and {}.',
                'Can you {} {} and {}?'
            ],
            3: [
                'Please {} {}, {}, and {}.',
                'I want to {} {}, {}, and {}.',
                'Can you {} {}, {}, and {}?'
            ],
        }

        self.shorten_labels = {
            'subway, metro, underground': 'metro',
            'police car (siren)': 'police siren',
            'bee, wasp, etc. buzzing': 'buzzing',
            'vehicle horn, car horn, honking': 'car horn',
            'rowboat, canoe, kayak rowing': 'rowboat'
        }

        self.shuffle = shuffle
        self.random = random

        print(f'Initialized {str(self.__class__.__name__)}: ')
        print(f'shuffle: {str(shuffle)} random: {str(random)}')

    def __call__(self, acts, spks, labels, ret_n_src=False):

        srcs = []
        for spk in spks:
            srcs.append(spk)
        for label in labels:
            if label in self.shorten_labels:
                src = 'the {} sound'.format(self.shorten_labels[label])
            else:
                src = 'the {} sound'.format(label)
            srcs.append(src)

        assert len(acts) == len(srcs)

        # For editing including volume up and down, 
        # skip the instruction of 'no change'.
        if 'D' in acts or 'U' in acts:
            phases = []
            for act, src in zip(acts, srcs):
                if act == '1': # no change
                    continue
                else:
                    verb = random.choice(self.lookups[act]) \
                    if self.random else self.lookups[act][0]
                    phase = verb.format(src)
                    phases.append(phase)

            if self.shuffle:
                random.shuffle(phases)

            if self.random:
                prompt = random.choice(self.templates[len(phases)]).format(*phases)
            else:
                prompt = self.templates[len(phases)][0].format(*phases)

            n_src = len(phases)

        # For extraction/removal-only problem, choose between
        # saying extracting targets or removing non-targets.
        # e.g. "Please extract 1..., 2..., and 3..."
        # and "Please remove 4..." are equivalent.
        else:
            e_or_r = random.choice(['1', '0'])
            verb = random.choice(self.lookups_extract_or_remove[e_or_r]) \
            if self.random else self.lookups_extract_or_remove[e_or_r][0]
            tars = [] # all sources we want to extract or remove
            for act, src in zip(acts, srcs):
                if act == e_or_r:
                    tars.append(src)

            if self.shuffle:
                random.shuffle(tars)
                
            if self.random:
                prompt = random.choice(self.templates_extract_or_remove[len(tars)]).format(verb, *tars)
            else:
                prompt = self.templates_extract_or_remove[len(tars)][0].format(verb, *tars)

            n_src = len(tars)

        if ret_n_src:
            return prompt, n_src
        else:
            return prompt


special_prompts = {
    # Speech enhancement
    '1100': [
        "Extract all speakers.",
        "Enhance the audio by removing background noises.",
        "Extract all human voices from the recording.",
        "Eliminate any non-speech sounds in the surroundings.",
        "Make the conversation as clean as possible."
    ],
    # Speech removal
    '0011': [
        "Remove all speakers.",
        "Extract the environmental sounds surrounding the audio.",
        "Filter and only keep the background sounds.",
        "Eliminate all human voices.",
        "Suppress the conversation."
    ],
    # Overall volume control
    'UUUU': [
        "Increase the volume of the audio.",
        "Amplify all sounds in the mixture.",
        "Raise the sound level of the audio.",
        "I can't hear. Make the recording louder.",
        "Up the audio volume."
    ],
    'DDDD': [
        "Decrease the volume of the audio.",
        "Make everything quieter.",
        "Lower the sound level of the audio.",
        "It's too loud. Turn down the volume of the recording.",
        "Down the audio volume."
    ],
    # Foreground/Background volume control
    'UU11': [
        "Increase the volume of the speeches.", 
        "Boost the voice volume of the talkers.", 
        "Can you amplify the sound of people speaking?",
        "Add some extra decibels to the conversation.",
        "Could you make the talking part a bit louder?"
    ],
    'DD11': [
        "Decrease the volume of the speeches.",
        "Could you lower the sound of people talking?",
        "Let's bring down the volume of human voices.",
        "Mind dialing back the talking level a bit?",
        "Shall we quiet down the speech?"
    ],
    '11UU': [
        "Increase the volume of the background sounds.",
        "Could you amplify the ambient sounds a bit more?",        
        "Let's make the surroundings a bit louder.", 
        "Can you boost the audio for non-verbal sounds?"
        "Make those other sounds a bit louder, leaving speech as is.",
    ],
    '11DD': [
        "Decrease the volume of the background sounds.",
        "Could you lower the level of ambient noises?",
        "Is it possible to reduce the intensity of the surrounding sounds?",
        "Lower the volume of the background disturbances.",
        "Can you reduce the background noises?"
    ],
    'UUDD': [
        "Increase the volume of the speeches and decrease the volume of the background sounds.",
        "Pump up the volume on the talks and reduce the noise in the environment.",
        "Enhance the conversation, and also quieten down those distracting background sounds.",
        "Decrease the volume of the background sounds and Increase the volume of the speakers.",
        "Reduce the background audio and turn up the volume on the talking parts."
    ],
    'DDUU': [
        "Decrease the volume of the speeches and increase the volume of the background sounds.",
        "Lower the speech volume and amplify the surrounding sounds.",
        "Could you quieten the speakers and elevate the environment sounds?",
        "Increase the volume of the background sounds and decrease the volume of the speeches.",
        "Add more volume to the surrounding, and decrease the speech volume."
    ],
    # Removal + Foreground/Background volume control
    'UU00': [
        "Extract all speakers and make them louder.",
        "Extract and boost all human voices in the mixture.",
        "Bring out and amplify the conversation in the recording.",
        "Could you increase the volume of the human voices present and remove the rest?",
        "Eliminate all background noises and turn up the volume of the speeches."
    ],
    '00UU': [
        "Extract the background sounds and make them louder.",
        "Extract and boost the surroundings in the mixture.",
        "Bring out and amplify the ambient sounds in the recording.",
        "Could you increase the volume of the non-speech sounds present and remove the rest?",
        "Eliminate all speakers and turn up the volume of the environment."
    ],
    'DD00': [
        "Extract all speakers but make them quieter.",
        "Extract and reduce the volume of all human voices in the mixture.",
        "Bring out but quieten the conversation in the recording.",
        "Could you decrease the volume of the human voices present and remove the rest?",
        "Eliminate all background noises and turn down the volume of the speeches."
    ],
    '00DD': [
        "Extract the background sounds but make them quieter.",
        "Extract and reduce the volume of the surroundings in the mixture.",
        "Bring out but quieten the ambient sounds in the recording.",
        "Could you decrease the volume of the non-speech sounds present and remove the rest?",
        "Eliminate all speakers and turn down the volume of the environment."
    ]
}


## Not used

def build_style_description(gender, pitch, tempo, energy, emotion):
    
    gender = 'female' if gender == 'F' else 'male'

    if emotion != 'neutral':
        description = 'the {} {} speaker characterized by {} pitch, {} tempo, and {} energy'\
            .format(emotion, gender, pitch, tempo, energy)
        
    else:
        description = 'the {} speaker characterized by {} pitch, {} tempo, and {} energy'\
            .format(gender, pitch, tempo, energy)
        
    return description