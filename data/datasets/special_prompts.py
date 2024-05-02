special_prompts_2speech_2audio = {
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

special_prompts_3speech_1audio = {
    # Speech enhancement
    '1110': [
        "Extract all speakers.",
        "Enhance the audio by removing background noises.",
        "Extract all human voices from the recording.",
        "Eliminate any non-speech sounds in the surroundings.",
        "Make the conversation as clean as possible."
    ],
    # Speech removal
    '0001': [
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
    'UUU1': [
        "Increase the volume of the speeches.", 
        "Boost the voice volume of the talkers.", 
        "Can you amplify the sound of people speaking?",
        "Add some extra decibels to the conversation.",
        "Could you make the talking part a bit louder?"
    ],
    'DDD1': [
        "Decrease the volume of the speeches.",
        "Could you lower the sound of people talking?",
        "Let's bring down the volume of human voices.",
        "Mind dialing back the talking level a bit?",
        "Shall we quiet down the speech?"
    ],
    '111U': [
        "Increase the volume of the background sounds.",
        "Could you amplify the ambient sounds a bit more?",        
        "Let's make the surroundings a bit louder.", 
        "Can you boost the audio for non-verbal sounds?"
        "Make those other sounds a bit louder, leaving speech as is.",
    ],
    '111D': [
        "Decrease the volume of the background sounds.",
        "Could you lower the level of ambient noises?",
        "Is it possible to reduce the intensity of the surrounding sounds?",
        "Lower the volume of the background disturbances.",
        "Can you reduce the background noises?"
    ],
    'UUUD': [
        "Increase the volume of the speeches and decrease the volume of the background sounds.",
        "Pump up the volume on the talks and reduce the noise in the environment.",
        "Enhance the conversation, and also quieten down those distracting background sounds.",
        "Decrease the volume of the background sounds and Increase the volume of the speakers.",
        "Reduce the background audio and turn up the volume on the talking parts."
    ],
    'DDDU': [
        "Decrease the volume of the speeches and increase the volume of the background sounds.",
        "Lower the speech volume and amplify the surrounding sounds.",
        "Could you quieten the speakers and elevate the environment sounds?",
        "Increase the volume of the background sounds and decrease the volume of the speeches.",
        "Add more volume to the surrounding, and decrease the speech volume."
    ],
    # Removal + Foreground/Background volume control
    'UUU0': [
        "Extract all speakers and make them louder.",
        "Extract and boost all human voices in the mixture.",
        "Bring out and amplify the conversation in the recording.",
        "Could you increase the volume of the human voices present and remove the rest?",
        "Eliminate all background noises and turn up the volume of the speeches."
    ],
    '000U': [
        "Extract the background sounds and make them louder.",
        "Extract and boost the surroundings in the mixture.",
        "Bring out and amplify the ambient sounds in the recording.",
        "Could you increase the volume of the non-speech sounds present and remove the rest?",
        "Eliminate all speakers and turn up the volume of the environment."
    ],
    'DDD0': [
        "Extract all speakers but make them quieter.",
        "Extract and reduce the volume of all human voices in the mixture.",
        "Bring out but quieten the conversation in the recording.",
        "Could you decrease the volume of the human voices present and remove the rest?",
        "Eliminate all background noises and turn down the volume of the speeches."
    ],
    '000D': [
        "Extract the background sounds but make them quieter.",
        "Extract and reduce the volume of the surroundings in the mixture.",
        "Bring out but quieten the ambient sounds in the recording.",
        "Could you decrease the volume of the non-speech sounds present and remove the rest?",
        "Eliminate all speakers and turn down the volume of the environment."
    ]
}

special_prompts_1speech_3audio = {
    # Speech enhancement
    '1000': [
        "Extract all speakers.",
        "Enhance the audio by removing background noises.",
        "Extract all human voices from the recording.",
        "Eliminate any non-speech sounds in the surroundings.",
        "Make the conversation as clean as possible."
    ],
    # Speech removal
    '0111': [
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
    'U111': [
        "Increase the volume of the speeches.", 
        "Boost the voice volume of the talkers.", 
        "Can you amplify the sound of people speaking?",
        "Add some extra decibels to the conversation.",
        "Could you make the talking part a bit louder?"
    ],
    'D111': [
        "Decrease the volume of the speeches.",
        "Could you lower the sound of people talking?",
        "Let's bring down the volume of human voices.",
        "Mind dialing back the talking level a bit?",
        "Shall we quiet down the speech?"
    ],
    '1UUU': [
        "Increase the volume of the background sounds.",
        "Could you amplify the ambient sounds a bit more?",        
        "Let's make the surroundings a bit louder.", 
        "Can you boost the audio for non-verbal sounds?"
        "Make those other sounds a bit louder, leaving speech as is.",
    ],
    '1DDD': [
        "Decrease the volume of the background sounds.",
        "Could you lower the level of ambient noises?",
        "Is it possible to reduce the intensity of the surrounding sounds?",
        "Lower the volume of the background disturbances.",
        "Can you reduce the background noises?"
    ],
    'UDDD': [
        "Increase the volume of the speeches and decrease the volume of the background sounds.",
        "Pump up the volume on the talks and reduce the noise in the environment.",
        "Enhance the conversation, and also quieten down those distracting background sounds.",
        "Decrease the volume of the background sounds and Increase the volume of the speakers.",
        "Reduce the background audio and turn up the volume on the talking parts."
    ],
    'DUUU': [
        "Decrease the volume of the speeches and increase the volume of the background sounds.",
        "Lower the speech volume and amplify the surrounding sounds.",
        "Could you quieten the speakers and elevate the environment sounds?",
        "Increase the volume of the background sounds and decrease the volume of the speeches.",
        "Add more volume to the surrounding, and decrease the speech volume."
    ],
    # Removal + Foreground/Background volume control
    'U000': [
        "Extract all speakers and make them louder.",
        "Extract and boost all human voices in the mixture.",
        "Bring out and amplify the conversation in the recording.",
        "Could you increase the volume of the human voices present and remove the rest?",
        "Eliminate all background noises and turn up the volume of the speeches."
    ],
    '0UUU': [
        "Extract the background sounds and make them louder.",
        "Extract and boost the surroundings in the mixture.",
        "Bring out and amplify the ambient sounds in the recording.",
        "Could you increase the volume of the non-speech sounds present and remove the rest?",
        "Eliminate all speakers and turn up the volume of the environment."
    ],
    'D000': [
        "Extract all speakers but make them quieter.",
        "Extract and reduce the volume of all human voices in the mixture.",
        "Bring out but quieten the conversation in the recording.",
        "Could you decrease the volume of the human voices present and remove the rest?",
        "Eliminate all background noises and turn down the volume of the speeches."
    ],
    '0DDD': [
        "Extract the background sounds but make them quieter.",
        "Extract and reduce the volume of the surroundings in the mixture.",
        "Bring out but quieten the ambient sounds in the recording.",
        "Could you decrease the volume of the non-speech sounds present and remove the rest?",
        "Eliminate all speakers and turn down the volume of the environment."
    ]
}

special_prompts_2speech_1audio = {
    # Speech enhancement
    '110': [
        "Extract all speakers.",
        "Enhance the audio by removing background noises.",
        "Extract all human voices from the recording.",
        "Eliminate any non-speech sounds in the surroundings.",
        "Make the conversation as clean as possible."
    ],
    # Speech removal
    '001': [
        "Remove all speakers.",
        "Extract the environmental sounds surrounding the audio.",
        "Filter and only keep the background sounds.",
        "Eliminate all human voices.",
        "Suppress the conversation."
    ],
    # Overall volume control
    'UUU': [
        "Increase the volume of the audio.",
        "Amplify all sounds in the mixture.",
        "Raise the sound level of the audio.",
        "I can't hear. Make the recording louder.",
        "Up the audio volume."
    ],
    'DDD': [
        "Decrease the volume of the audio.",
        "Make everything quieter.",
        "Lower the sound level of the audio.",
        "It's too loud. Turn down the volume of the recording.",
        "Down the audio volume."
    ],
    # Foreground/Background volume control
    'UU1': [
        "Increase the volume of the speeches.", 
        "Boost the voice volume of the talkers.", 
        "Can you amplify the sound of people speaking?",
        "Add some extra decibels to the conversation.",
        "Could you make the talking part a bit louder?"
    ],
    'DD1': [
        "Decrease the volume of the speeches.",
        "Could you lower the sound of people talking?",
        "Let's bring down the volume of human voices.",
        "Mind dialing back the talking level a bit?",
        "Shall we quiet down the speech?"
    ],
    '11U': [
        "Increase the volume of the background sounds.",
        "Could you amplify the ambient sounds a bit more?",        
        "Let's make the surroundings a bit louder.", 
        "Can you boost the audio for non-verbal sounds?"
        "Make those other sounds a bit louder, leaving speech as is.",
    ],
    '11D': [
        "Decrease the volume of the background sounds.",
        "Could you lower the level of ambient noises?",
        "Is it possible to reduce the intensity of the surrounding sounds?",
        "Lower the volume of the background disturbances.",
        "Can you reduce the background noises?"
    ],
    'UUD': [
        "Increase the volume of the speeches and decrease the volume of the background sounds.",
        "Pump up the volume on the talks and reduce the noise in the environment.",
        "Enhance the conversation, and also quieten down those distracting background sounds.",
        "Decrease the volume of the background sounds and Increase the volume of the speakers.",
        "Reduce the background audio and turn up the volume on the talking parts."
    ],
    'DDU': [
        "Decrease the volume of the speeches and increase the volume of the background sounds.",
        "Lower the speech volume and amplify the surrounding sounds.",
        "Could you quieten the speakers and elevate the environment sounds?",
        "Increase the volume of the background sounds and decrease the volume of the speeches.",
        "Add more volume to the surrounding, and decrease the speech volume."
    ],
    # Removal + Foreground/Background volume control
    'UU0': [
        "Extract all speakers and make them louder.",
        "Extract and boost all human voices in the mixture.",
        "Bring out and amplify the conversation in the recording.",
        "Could you increase the volume of the human voices present and remove the rest?",
        "Eliminate all background noises and turn up the volume of the speeches."
    ],
    '00U': [
        "Extract the background sounds and make them louder.",
        "Extract and boost the surroundings in the mixture.",
        "Bring out and amplify the ambient sounds in the recording.",
        "Could you increase the volume of the non-speech sounds present and remove the rest?",
        "Eliminate all speakers and turn up the volume of the environment."
    ],
    'DD0': [
        "Extract all speakers but make them quieter.",
        "Extract and reduce the volume of all human voices in the mixture.",
        "Bring out but quieten the conversation in the recording.",
        "Could you decrease the volume of the human voices present and remove the rest?",
        "Eliminate all background noises and turn down the volume of the speeches."
    ],
    '00D': [
        "Extract the background sounds but make them quieter.",
        "Extract and reduce the volume of the surroundings in the mixture.",
        "Bring out but quieten the ambient sounds in the recording.",
        "Could you decrease the volume of the non-speech sounds present and remove the rest?",
        "Eliminate all speakers and turn down the volume of the environment."
    ]
}

special_prompts_1speech_2audio = {
    # Speech enhancement
    '100': [
        "Extract all speakers.",
        "Enhance the audio by removing background noises.",
        "Extract all human voices from the recording.",
        "Eliminate any non-speech sounds in the surroundings.",
        "Make the conversation as clean as possible."
    ],
    # Speech removal
    '011': [
        "Remove all speakers.",
        "Extract the environmental sounds surrounding the audio.",
        "Filter and only keep the background sounds.",
        "Eliminate all human voices.",
        "Suppress the conversation."
    ],
    # Overall volume control
    'UUU': [
        "Increase the volume of the audio.",
        "Amplify all sounds in the mixture.",
        "Raise the sound level of the audio.",
        "I can't hear. Make the recording louder.",
        "Up the audio volume."
    ],
    'DDD': [
        "Decrease the volume of the audio.",
        "Make everything quieter.",
        "Lower the sound level of the audio.",
        "It's too loud. Turn down the volume of the recording.",
        "Down the audio volume."
    ],
    # Foreground/Background volume control
    'U11': [
        "Increase the volume of the speeches.", 
        "Boost the voice volume of the talkers.", 
        "Can you amplify the sound of people speaking?",
        "Add some extra decibels to the conversation.",
        "Could you make the talking part a bit louder?"
    ],
    'D11': [
        "Decrease the volume of the speeches.",
        "Could you lower the sound of people talking?",
        "Let's bring down the volume of human voices.",
        "Mind dialing back the talking level a bit?",
        "Shall we quiet down the speech?"
    ],
    '1UU': [
        "Increase the volume of the background sounds.",
        "Could you amplify the ambient sounds a bit more?",        
        "Let's make the surroundings a bit louder.", 
        "Can you boost the audio for non-verbal sounds?"
        "Make those other sounds a bit louder, leaving speech as is.",
    ],
    '1DD': [
        "Decrease the volume of the background sounds.",
        "Could you lower the level of ambient noises?",
        "Is it possible to reduce the intensity of the surrounding sounds?",
        "Lower the volume of the background disturbances.",
        "Can you reduce the background noises?"
    ],
    'UDD': [
        "Increase the volume of the speeches and decrease the volume of the background sounds.",
        "Pump up the volume on the talks and reduce the noise in the environment.",
        "Enhance the conversation, and also quieten down those distracting background sounds.",
        "Decrease the volume of the background sounds and Increase the volume of the speakers.",
        "Reduce the background audio and turn up the volume on the talking parts."
    ],
    'DUU': [
        "Decrease the volume of the speeches and increase the volume of the background sounds.",
        "Lower the speech volume and amplify the surrounding sounds.",
        "Could you quieten the speakers and elevate the environment sounds?",
        "Increase the volume of the background sounds and decrease the volume of the speeches.",
        "Add more volume to the surrounding, and decrease the speech volume."
    ],
    # Removal + Foreground/Background volume control
    'U00': [
        "Extract all speakers and make them louder.",
        "Extract and boost all human voices in the mixture.",
        "Bring out and amplify the conversation in the recording.",
        "Could you increase the volume of the human voices present and remove the rest?",
        "Eliminate all background noises and turn up the volume of the speeches."
    ],
    '0UU': [
        "Extract the background sounds and make them louder.",
        "Extract and boost the surroundings in the mixture.",
        "Bring out and amplify the ambient sounds in the recording.",
        "Could you increase the volume of the non-speech sounds present and remove the rest?",
        "Eliminate all speakers and turn up the volume of the environment."
    ],
    'D00': [
        "Extract all speakers but make them quieter.",
        "Extract and reduce the volume of all human voices in the mixture.",
        "Bring out but quieten the conversation in the recording.",
        "Could you decrease the volume of the human voices present and remove the rest?",
        "Eliminate all background noises and turn down the volume of the speeches."
    ],
    '0DD': [
        "Extract the background sounds but make them quieter.",
        "Extract and reduce the volume of the surroundings in the mixture.",
        "Bring out but quieten the ambient sounds in the recording.",
        "Could you decrease the volume of the non-speech sounds present and remove the rest?",
        "Eliminate all speakers and turn down the volume of the environment."
    ]
}

special_prompts_2speech = {
    # Overall volume control
    'UU': [
        "Increase the volume of the audio.",
        "Amplify all sounds in the mixture.",
        "Raise the sound level of the audio.",
        "I can't hear. Make the recording louder.",
        "Up the audio volume."
    ],
    'DD': [
        "Decrease the volume of the audio.",
        "Make everything quieter.",
        "Lower the sound level of the audio.",
        "It's too loud. Turn down the volume of the recording.",
        "Down the audio volume."
    ]
}

special_prompts_2audio = {
    # Overall volume control
    'UU': [
        "Increase the volume of the audio.",
        "Amplify all sounds in the mixture.",
        "Raise the sound level of the audio.",
        "I can't hear. Make the recording louder.",
        "Up the audio volume."
    ],
    'DD': [
        "Decrease the volume of the audio.",
        "Make everything quieter.",
        "Lower the sound level of the audio.",
        "It's too loud. Turn down the volume of the recording.",
        "Down the audio volume."
    ]
}


