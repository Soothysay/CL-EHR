# CL-EHR

This is the official code repository of the paper titled "Continually-Adaptive Representation Learning Framework for Time-Sensitive Healthcare Applications" appearing in CIKM 2023.
![alt text](https://github.com/soothysay/cl-ehr/blob/main/model_figure.PNG?raw=true)
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

### Step 3

1. To get the clinical note embeddings, pre-train the BERT model by training over multiple GPUs using the script `BERT_multi_GPU.py` file.
2. Then store the model state_dict for the main training script.

 
### Step 4

1. For the first period, train the model by using the script `main.py` setting decent_pretrain_epochs=200. for the first period, comment out the CL loss in lines 1199,1173 and 1174.
2. Store the model embeddings for each epoch/ each 10 epochs.
3. For other periods, set decent_pretrain_epochs=0 and train the model. Subsequently, the CL loss is evaaluated on the previous model's parameters. So, modify that with the appropriate locations.
4. t-batching size can be increased. This implementation backpropagates on each t-batch due to memory constraints. However, performance will be better if you take more t-batches before backpropagation.

### Step 5

1. Evaluate the performance of the model in each downstream task by taking the resultant patient embeddings on the binary task and modifying the locations in the file `CDI_LR.py`.
