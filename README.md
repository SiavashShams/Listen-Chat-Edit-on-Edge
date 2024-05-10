<!---
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/UHGdSN-p)
# E6692 Spring 2024: Final Project

The description of the final project for e6692 2024 spring is provided in [the Google drive](https://docs.google.com/document/d/1o4E8MksTKACW7tfuazcWeNCaSGLv9YKAd7A5lqFmzkA/edit?usp=sharing)

This repo is an (empty) template which is distributed to students as a part of the Github classroom assignment process.

The repository (assigned by the Github Classroom) will become (after students accept the github classroom link invite) the repository for all project contents.

## How to modify this README.md file
Students need to maintain this repo to have a "professional look":
* Remove the instructions (this text)
* Provide description of the topic/project
* Provide organization of this repo 
* Add all relevant links: name of Google docs and link to them, links to public repos used, etc.
* For paper reviews it should include the organization of the directory, brief description how to run the code, what is in the code, links to relevant papers, links to relevant githubs, etc...

## INSTRUCTIONS for (re) naming the student's repository for the final project with one student:
* Students need to use the following naming rules for the repository with their solutions: e6692-2024Spring-FinalProject-GroupID-UNI 
(the first part "e6692-2024Spring-FinalProject" will probably be inherited from the assignment, so only UNI needs to be added) 
* Initially, the system may give the repo a name which ends with a student's Github userid. 
The student must change that name and replace it with the name requested in the point above (containing their UNI)
* Good Example: e6692-2024Spring-FinalProject-GroupID-zz9999;   Bad example: e6692-2024Spring-FinalProject-ee6040-2024Spring-FinalProject-GroupID-zz9999.
* This change can be done from the "Settings" tab which is located on the repo page.

## INSTRUCTIONS for naming the students' repository for the final project with more students. 
(Students need to use a 4-letter groupID/teamID): 
* Template: e6692-2024Spring-FinalProject-GroupID-UNI1-UNI2-UNI3. -> Example: e6692-2024Spring-FinalProject-MEME-zz9999-aa9999-aa0000.
-->
# Listen, Chat, And Edit on Edge: Text-Guided Soundscape Modification for Real-Time Auditory Experience

### What is this project about?
**Listen, Chat, and Edit (LCE)** is a cutting-edge multimodal sound mixture editor designed to modify each sound source in a mixture based on user-provided text instructions. The system features a user-friendly chat interface and the unique ability to edit multiple sound sources simultaneously within a mixture without the need for separation. Using open-vocabulary text prompts interpreted by a large language model, LCE creates a semantic filter to edit sound mixtures, which are then decomposed, filtered, and reassembled into the desired output.


## Project Structure
- **data/datasets**: Contains the scripts used to process dataset and prompts.
- **demonstration**: A demonstration of an input mixure and the edited version.
- **embeddings**: The pkl file recieved from the LLM are stored in this folder.
- **hparams**: Hyperparameters settings for the models.
- **llm_cloud**: Configuration and scripts for cloud-based language model interactions.
- **modules**: Core modules and utilities for the project.
- **prompts**: Handling and processing of text prompts.
- **pubsub**: Setup for publish-subscribe messaging patterns.
- **utils**: Utility scripts for general purposes.
- **E6692.2022Spring.LCEE.ss6928.pkk2125.presentationFinal.pptx**: Final presentation file detailing project overview and results.
- **profiling.ipynb**: Jupyter notebook for profiling the modules in terms of inference speed and gpu memory usage.
- **run_lce.ipynb**: Main executable notebook for the LCE system.
- **run_prompt_reader.ipynb**: Notebook for reading and processing prompts.
- **run_prompt_reader_profiling.ipynb**: Profiling for the prompt reader.
- **run_sound_editor_nosb.ipynb**: Notebook for the sound editor module without SpeechBrain.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/eecse6692/e6692-2024spring-finalproject-lcee.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the main LCE application:
```
run_lce.ipynb
```
For a demonstration of the system's capabilities, refer to the `demonstration` folder.

### Implementation

- Deploy Conv-TasNet on the Jetson Nano.
- Deploy LLAMA 2 on a GCP server
- Send a prompt to the server. Communication is handled in two methods - one, through SSH and the other, through Pub/Sub service.
- LLM computed the embedding and publishes back the embedding, which is input to the Conv-TasNet model.
- The resulting audio mixture is ready to be played!




### References
[Listen, Chat, And Edit](https://arxiv.org/pdf/2402.03710)

