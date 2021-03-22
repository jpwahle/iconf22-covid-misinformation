'''
Author: Nischal A, B.Tech Computer Science, IIT Patna
'''

################ Setup ##############
#####################################
import torch
import os

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.   
    # gpu_number = 3 
    # gpu_name = "cuda:" + str(gpu_number)
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    
    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

################ All Imports ##############
###########################################


import time
import datetime
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import transformers
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW, get_linear_schedule_with_warmup


###### Main LM Classifier Model ######
######################################

class SentencePairClassifier(nn.Module):

    def __init__(self, bert_model="bert-base-uncased", freeze_bert=False):
        super(SentencePairClassifier, self).__init__()
        self.model_name = bert_model
        #  Instantiating BERT-based model object
        if bert_model == "covid-roberta":
        	self.config = transformers.RobertaConfig.from_pretrained('/home1/tirthankar/Nischal/DKE/cord-19/models/COVID_RoBERTa/old/config.json')
        	self.bert_layer = transformers.RobertaModel.from_pretrained('/home1/tirthankar/Nischal/DKE/cord-19/models/COVID_RoBERTa/old/pytorch_model.bin', config = self.config)
        elif bert_model == "covid-bart":
        	self.config = transformers.BartConfig.from_pretrained("/home1/tirthankar/Nischal/DKE/cord-19/models/COVID_Bart_mlm/config.json", output_hidden_states=False)
	        self.bert_layer = transformers.BartModel.from_pretrained("/home1/tirthankar/Nischal/DKE/cord-19/models/COVID_Bart_mlm/pytorch_model.bin", config = self.config)
        elif bert_model == "covid-deberta":
        	self.config = transformers.DebertaConfig.from_pretrained("/home1/tirthankar/Nischal/DKE/cord-19/models/COVID_DeBERTa/config.json", output_hidden_states=False)
	        self.bert_layer = transformers.DebertaModel.from_pretrained("/home1/tirthankar/Nischal/DKE/cord-19/models/COVID_DeBERTa/pytorch_model.bin", config = self.config)
        elif bert_model == "covid-longformer":
        	self.config = transformers.LongformerConfig.from_pretrained("/home1/tirthankar/Nischal/DKE/cord-19/models/COVID_longformer/config.json", output_hidden_states=False)
	        self.bert_layer = transformers.LongformerModel.from_pretrained("/home1/tirthankar/Nischal/DKE/cord-19/models/COVID_longformer/pytorch_model.bin", config = self.config)
        elif bert_model == "covid-twitter-bert":
        	self.config = transformers.BertConfig.from_pretrained('/home1/tirthankar/Nischal/DKE/cord-19/models/COVID_TweetBERT/config.json')
        	self.bert_layer = transformers.BertModel.from_pretrained('/home1/tirthankar/Nischal/DKE/cord-19/models/COVID_TweetBERT/pytorch_model.bin', config = self.config)	    	
        else:
	        self.config = AutoConfig.from_pretrained(bert_model, output_hidden_states=False)
	        self.bert_layer = AutoModel.from_pretrained(bert_model, config = self.config)
        large_models = ["digitalepidemiologylab/covid-twitter-bert-v2", "covid-twitter-bert", "manueltonneau/biocovid-bert-large-cased"]
        if bert_model in large_models:
        	self.hidden_size = 1024
        else:
        	self.hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(self.hidden_size, 3)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        special_models = ['roberta-base', 'distilbert-base-uncased', 'covid-roberta', 'facebook/bart-base',
        					'covid-bart', 'covid-longformer', "vinai/bertweet-base"]
        if self.model_name in special_models:
        	token_type_ids = None
        if self.model_name == 'distilbert-base-uncased' or 'facebook/bart-base' or 'covid-bart':
        	hidden_state  = self.bert_layer(input_ids, attn_masks, token_type_ids)
        	# print(len(pooler_output))
        	# for obj in pooler_output:
        	# 	print(obj.shape)
        	pooler_output = hidden_state[0][:,0]
        else:
        	cont_reps, pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids)

        # Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a
        # Linear Layer and a Tanh activation. The Linear layer weights were trained from the sentence order prediction or next sentence prediction (BERT)
        # objective during pre-training.
        logits = self.cls_layer(self.dropout(pooler_output))

        return logits

################ Setting Seed ##############
###############################################
def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

################ Dataset Loading ##############
###############################################

def get_data(train_df, test_df):
	X1, X2, Y_train = train_df['title'], train_df['content'], train_df['label']
	x1_test, x2_test, y_test = test_df['title'], test_df['content'], test_df['label']

	VALIDATION_RATIO = 0.1
	RANDOM_STATE = 9527
	x1_train, x1_val, \
	x2_train, x2_val, \
	y_train, y_val = \
	    train_test_split(
	        X1, X2,
	        Y_train,
	        test_size=VALIDATION_RATIO, 
	        random_state=RANDOM_STATE
	)

	# Converting everything to list
	x1_train = x1_train.tolist()
	x2_train = x2_train.tolist()
	y_train = y_train.tolist()
	x1_val, x2_val, y_val = x1_val.tolist(), x2_val.tolist(), y_val.tolist()
	x1_test, x2_test, y_test = x1_test.tolist(), x2_test.tolist(), y_test.tolist()
	return (x1_train, x2_train, y_train), (x1_val, x2_val, y_val), (x1_test, x2_test, y_test)

################ Tokenizer ####################
###############################################
def tokenize(model_name, premise_data, hypothesis_data, tokenizer, MAX_LEN):
	print('Tokenizing')
	# add special tokens for BERT to work properly
	model_lst = ["covid-roberta", "covid-bart", "covid-longformer", "vinai/bertweet-base"]
	if model_name in model_lst:
		sentences = ["<s> " + premise_data[i] + " </s>" + hypothesis_data[i] + "</s>" for i in range(0,len(premise_data))]
	else:
		sentences = ["[CLS] " + premise_data[i] + " [SEP]" + hypothesis_data[i] + "[SEP]" for i in range(0,len(premise_data))]
	tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
	print ("Tokenize the first sentence:")
	print (tokenized_texts[0])
	# Pad our input tokens
	input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
	                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
	# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
	input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
	input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
	# Create attention masks
	attention_masks = []
	# Create a mask of 1s for each token followed by 0s for padding
	for seq in input_ids:
	  seq_mask = [float(i>0) for i in seq]
	  attention_masks.append(seq_mask)

	# Printing the input_ids
	print('Input_ids[0]', input_ids[0])
	print('Input_ids[0] elements')
	for i in input_ids[0]:
		print(i, type(i), end = " ")

	token_type_ids = []
	for seq in input_ids:
		type_id = []
		condition = 'sent1'
		for i in seq:
			if condition == 'sent1':
				type_id.append(0)
				if i == 102:
					condition = 'sent2'
			elif condition == 'sent2':
				type_id.append(1)
		token_type_ids.append(type_id)
	print(token_type_ids[0])
		

	# Finally convert this into torch tensors
	data_inputs = torch.tensor(input_ids, device =device)
	data_masks = torch.tensor(attention_masks, device =device)
	data_token_ids = torch.tensor(token_type_ids, device = device)
	return data_inputs, data_masks, data_token_ids

################ Data Loader ####################
###############################################
def get_data_loader(batch_size, inputs, masks, token_ids, labels):
	data = TensorDataset(inputs, masks, token_ids, labels)
	sampler = RandomSampler(data)
	dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
	return data, sampler, dataloader

################ Transformer Model ####################
#######################################################
def get_transformer_model(modelname):
	if modelname == 'covid-roberta':
		tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base', max_len=512)
		# tokenizer = transformers.RobertaTokenizer.from_pretrained("/home2/tirthankar/Nischal/dke/cord-19/cord-19/models/COVID", max_len=512)
	elif modelname == 'covid-bart':
		tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/bart-base', max_len=512)
	elif modelname == 'covid-deberta':
		tokenizer = transformers.AutoTokenizer.from_pretrained('microsoft/deberta-base', max_len=512)
	elif modelname == 'covid-longformer':
		tokenizer = transformers.AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
	elif modelname == 'covid-twitter-bert':
		tokenizer = transformers.AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', do_lower_case = True)
	else:
		tokenizer = AutoTokenizer.from_pretrained(modelname, do_lower_case = True) 

	model = SentencePairClassifier(bert_model = modelname, freeze_bert = False)
	# Tell pytorch to run this model on the GPU.
	model.cuda()

	return tokenizer, model

################ Optimizer Scheduler ####################
###############################################
def get_optimizer_scheduler(name, model, train_dataloader_len, epochs, lr_set):
	if name == "Adam":
		optimizer = AdamW(model.parameters(),
                  lr = lr_set, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
		total_steps = train_dataloader_len * epochs

		# Create the learning rate scheduler.
		scheduler = get_linear_schedule_with_warmup(optimizer, 
		                                            num_warmup_steps = 0, # Default value in run_glue.py
		                                            num_training_steps = total_steps)
	return optimizer, scheduler

################ Evaluating Loss ######################
#######################################################
def evaluate_loss(net, device, criterion, dataloader):
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, token_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_ids, labels = \
                seq.to(device), attn_masks.to(device), token_ids.to(device), labels.to(device)
            logits = net(seq, attn_masks, token_ids)
            mean_loss += criterion(logits.squeeze(-1), labels).item()
            count += 1

    return mean_loss / count

################ Flat Accuracy Calculation ####################
###############################################################

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

################ Validation Accuracy Calculation ####################
###############################################################

def evaluate_accuracy(model, device, validation_dataloader):
	model.eval()
	# Tracking variables 
	eval_loss, eval_accuracy = 0, 0
	nb_eval_steps, nb_eval_examples = 0, 0
	# Evaluate data for one epoch
	for batch in validation_dataloader:
	    # Add batch to GPU
	    batch = tuple(t.to(device) for t in batch)	    
	    # Unpack the inputs from our dataloader
	    b_input_ids, b_input_mask, b_token_ids, b_labels = batch	    
	    
	    # Telling the model not to compute or store gradients, saving memory and
	    # speeding up validation
	    with torch.no_grad(): 
	    	# Forward pass, calculate logit predictions.
	        # This will return the logits rather than the loss because we have
	        # not provided labels.
	        # token_type_ids is the same as the "segment ids", which 
	        # differentiates sentence 1 and 2 in 2-sentence tasks.
	        # The documentation for this `model` function is here: 
	        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
	    	logits = model(b_input_ids, b_input_mask, b_token_ids)       

	    # Move logits and labels to CPU
	    logits = logits.detach().cpu().numpy()
	    label_ids = b_labels.to('cpu').numpy()
	    
	    # Calculate the accuracy for this batch of test sentences.
	    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
	    
	    # Accumulate the total accuracy.
	    eval_accuracy += tmp_eval_accuracy

	    # Track the number of batches
	    nb_eval_steps += 1
	accuracy = eval_accuracy/nb_eval_steps
	return accuracy

################ Evaluation Code ##############################
###############################################################

def evaluate(prediction_dataloader, model, model_name, path_to_model, load = False):
	# Prediction on test set
	if load:
		print("Loading the weights of the model...")
		model.load_state_dict(torch.load(path_to_model))

	print('Evaluating on the testset')

	# Put model in evaluation mode
	model.eval()

	# Tracking variables 
	predictions , true_labels = [], []

	# Predict 
	for batch in prediction_dataloader:
	  # Add batch to GPU
	  batch = tuple(t.to(device) for t in batch)
	  
	  # Unpack the inputs from our dataloader
	  b_input_ids, b_input_mask, b_token_ids, b_labels = batch
	  
	  # Telling the model not to compute or store gradients, saving memory and 
	  # speeding up prediction
	  with torch.no_grad():
	      # Forward pass, calculate logit predictions
	      logits = model(b_input_ids, b_input_mask, b_token_ids)

	  # Move logits and labels to CPU
	  logits = logits.detach().cpu().numpy()
	  label_ids = b_labels.to('cpu').numpy()

	  pred_flat = np.argmax(logits, axis=1).flatten()
	  labels_flat = label_ids.flatten()
	  
	  # Store predictions and true labels
	  predictions.extend(pred_flat)
	  true_labels.extend(labels_flat)

	print('DONE.')

	# Code for result display
	print('Coaid classification accuracy is')
	print(metrics.accuracy_score(true_labels, predictions)*100)
	print(classification_report(true_labels, predictions, target_names = ['fake', 'real']))
	# Converting to csv
	# Removed transpose - check if actually required
	bert_model = model_name.split('/')
	if len(bert_model) > 1:
		bert_model = bert_model[1]
	else:
		bert_model = bert_model[0]
	clsf_report = pd.DataFrame(classification_report(y_true = true_labels, y_pred = predictions, output_dict=True, target_names = ['fake', 'real']))
	clsf_report.to_csv(str('resultscoaid/'+bert_model+'_coaid.csv'), index= True)


################ Main TRAINING CODE ###########################
###############################################################
def train_bert(net, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate, bert_model):
    best_loss = np.Inf
    best_ep = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5  # print the training loss 5 times per epoch
    iters = []
    train_losses = []
    val_losses = []

    for ep in range(epochs):

        net.train()
        running_loss = 0.0
        for it, (seq, attn_masks, token_ids, labels) in enumerate(tqdm(train_loader)):

            # Converting to cuda tensors
            seq, attn_masks, token_ids, labels = \
                seq.to(device), attn_masks.to(device), token_ids.to(device), labels.to(device)
    		
            # Obtaining the logits from the model
            logits = net(seq, attn_masks, token_ids)

            # Computing loss
            loss = criterion(logits.squeeze(-1), labels)
            loss = loss / iters_to_accumulate  # Normalize the loss because it is averaged

            # Backpropagating the gradients
            # Calls backward()
            loss.backward()

            if (it + 1) % iters_to_accumulate == 0:
                # Optimization step
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, opti.step() is then called,
                # otherwise, opti.step() is skipped.
                opti.step()
                # Adjust the learning rate based on the number of iterations.
                lr_scheduler.step()
                # Clear gradients
                net.zero_grad()


            running_loss += loss.item()

            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                print("Iteration {}/{} of epoch {} complete. Loss : {} "
                      .format(it+1, nb_iterations, ep+1, running_loss / print_every))

                running_loss = 0.0


        val_loss = evaluate_loss(net, device, criterion, val_loader)  # Compute validation loss
        val_accuracy = evaluate_accuracy(net, device, val_loader)
        print()
        print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))
        print("Epoch {} complete! Validation Accuracy : {}".format(ep+1, val_accuracy))

        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            print()
            net_copy = copy.deepcopy(net)  # save a copy of the model
            best_loss = val_loss
            best_ep = ep + 1

    # Saving the model
    bert_model = bert_model.split('/')
    if len(bert_model) > 1:
    	bert_model = bert_model[1]
    else:
    	bert_model = bert_model[0]
    path_to_model='models/{}_lr_{}_val_loss_{}_ep_{}.pt'.format(bert_model, lr, round(best_loss, 5), best_ep)
    torch.save(net_copy.state_dict(), path_to_model)
    print("The model has been saved in {}".format(path_to_model))

    del loss
    torch.cuda.empty_cache()
    return net

################ Main Function ####################
###################################################
def main():
	#  Set all seeds to make reproducible results
	set_seed(1)
	print("Welcome to LM Pytorch Sentence Classification Pipeline")
	
	# Loading the datasets
	print('Loading CoAid dataset')
	train_df = pd.read_csv('datasets/UnbalTrain_CoAid_News.csv')
	print(train_df.columns)

	# Test set
	test_df = pd.read_csv('datasets/UnbalTest_CoAid_News.csv')
	print(test_df.columns)

	(x1_train, x2_train, y_train), (x1_val, x2_val, y_val), (x1_test, x2_test, y_test) = get_data(train_df, test_df)

	# Geting the Transformer Tokenized Output
	MAX_LEN=200
	#model_name = 'bert-base-uncased'
	#model_name = 'distilbert-base-uncased'
	#model_name = 'roberta-base'
	#model_name = 'allenai/scibert_scivocab_uncased'
	#model_name = 'emilyalsentzer/Bio_ClinicalBERT'
	#model_name = 'ttumyche/bluebert' #pytorch version fault
	#model_name = 'deepset/covid_bert_base'
	#model_name = 'lordtt13/COVID-SciBERT'
	#model_name = 'facebook/bart-base'
	#model_name = 'vinai/bertweet-base'
	#model_name = 'digitalepidemiologylab/covid-twitter-bert-v2'
	#model_name = 'manueltonneau/clinicalcovid-bert-base-cased'
	#model_name = 'manueltonneau/biocovid-bert-large-cased'
	#model_name = 'covid-roberta'
	#model_name = 'covid-bart'
	#model_name = 'covid-deberta'
	#model_name = 'covid-longformer'
	model_name = 'covid-twitter-bert'
	print('Using modelname', model_name)
	#batch_size = 4
	if model_name == 'digitalepidemiologylab/covid-twitter-bert-v2' or 'covid-twitter-bert':
		batch_size = 8
	elif model_name == 'covid-longformer':
		batch_size = 4
	else:
		batch_size = 16
	epochs = 3
	lr = 2e-5
	iters_to_accumulate = 2
	tokenizer, model = get_transformer_model(model_name)
	# Parallelizing the model
	model = torch.nn.DataParallel(model)
	print("Successfully retrived tokenizer and the model!")
	train_inputs, train_masks, train_token_ids = tokenize(model_name, x1_train, x2_train, tokenizer, MAX_LEN)
	val_inputs, val_masks, val_token_ids = tokenize(model_name, x1_val, x2_val, tokenizer, MAX_LEN)
	test_inputs, test_masks, test_token_ids = tokenize(model_name, x1_test, x2_test, tokenizer, MAX_LEN)

	# Converting the labels into torch tensors
	train_labels = torch.tensor(y_train, dtype=torch.long, device =device)
	val_labels = torch.tensor(y_val, dtype=torch.long, device =device)
	test_labels = torch.tensor(y_test, dtype=torch.long, device =device)

	# Printing the shape of these tensors
	print('Printing the shape of the final tensors')
	print('Train input', train_inputs.shape, 'Train Masks', train_masks.shape, 'Train Token_ids', train_token_ids.shape, 'Train Labels', train_labels.shape)
	print('Val input', val_inputs.shape, 'Val Masks', val_inputs.shape, 'Val Labels', val_inputs.shape)
	print('Test input', test_inputs.shape, 'Test Masks', test_inputs.shape, 'Test Labels', test_inputs.shape)
	# Getting the dataloaders
	train_data, train_sampler, train_dataloader = get_data_loader(batch_size, train_inputs, train_masks, train_token_ids, train_labels)
	val_data, val_sampler, val_dataloader = get_data_loader(batch_size, val_inputs, val_masks, val_token_ids, val_labels)
	test_data, test_sampler, test_dataloader = get_data_loader(batch_size, test_inputs, test_masks, test_token_ids, test_labels)
	print("Successfull in data prepration!")

	# Getting optimzer and scheduler
	optimizer, scheduler = get_optimizer_scheduler("Adam", model, len(train_dataloader), epochs, lr)
	print("Successfully loaded optimzer and scheduler")

	# Main Traning
	criterion = nn.CrossEntropyLoss()
	model = train_bert(model, criterion, optimizer, lr, scheduler, train_dataloader, val_dataloader, epochs, iters_to_accumulate, model_name)
	
	# Evaluation on Test set
	path_to_model = 'models/covid-twitter-bert_lr_2e-05_val_loss_0.04654_ep_2.pt'
	evaluate(test_dataloader, model, model_name, path_to_model = path_to_model, load = False)

if __name__ == "__main__":
    main()