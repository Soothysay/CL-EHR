# CL-EHR

This is the official code repository of the paper titled "Continually-Adaptive Representation Learning Framework for Time-Sensitive Healthcare Applications" appearing in CIKM 2023.

## Instructions

To run the scripts for CL-EHR, follow these steps:

### Step 1

1. Create files for patient dynamic and static features for each timestamp and records of patient interactions with entities.
2. Create a file containing the following schema:
   - user_id (pid)
   - item_id (entity id)
   - timestamp
   - state_label
   - label (indicating the type of interaction)
   - patient features

### Step 2

For clinical notes:

1. Create a file of clinical notes interacting with each patient over each timestamp.
2. Get the dynamicW2V embeddings for words in the notes using the scripts in the `/preprocessing` directory.
3. Append the file with the file created in Step 1, with each note having a unique item_id.
4. The label of the interaction will be 'N'.

You can refer to the code in this repository for more details on how to implement these steps for your project.
