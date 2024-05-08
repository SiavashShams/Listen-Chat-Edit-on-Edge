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
# Listen, Chat, And Edit
## Text-Guided Soundscape Modification For Enhanced Auditory Experience

### What is this project about?
An audio clip can be though of as a linear combination of its constituent sound mixtures, each with a unique semantic descriptor - gender, pitch, tempo, energy, emotion. The idea is to try to edit an audio sample by attenuating each component in the mixture. The amount of attenuation is prompted to the model with a text input that is decoded by an LLM. The decoded embedding produced by the LLM is input to a sound mixture model that attends to each mixture as described by the text promt. The output is an audio which is a combination of the attenuated sound mixtures.

### Implementation

- Deploy Conv-TasNet on the Jetson Nano.
- Deploy LLAMA 2 on a GCP server
- Send a prompt to the server. Communication is handled in two methods - one, through SSH and the other, through Pub/Sub service.
- LLM computed the embedding and publishes back the embedding, which is input to the Conv-TasNet model.
- The resulting audio mixture is ready to be played!




### References
[Listen, Chat, And Edit](https://arxiv.org/pdf/2402.03710)

