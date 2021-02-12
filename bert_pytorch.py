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
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup


################ Dataset Loading ##############
###############################################

def get_data(train_df, test_df):
	X1, X2, Y_train = train_df['premise'], train_df['text'], train_df['stance']
	x1_test, x2_test, y_test = test_df['premise'], test_df['text'], test_df['stance']

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
def tokenize(premise_data, hypothesis_data, tokenizer, MAX_LEN):
	print('Tokenizing')
	# add special tokens for BERT to work properly
	sentences = ["[CLS] " + premise_data[i] + " [SEP]" + hypothesis_data[i] for i in range(0,len(premise_data))]
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

	# Finally convert this into torch tensors
	data_inputs = torch.tensor(input_ids, device =device)
	data_masks = torch.tensor(attention_masks, device =device)
	return data_inputs, data_masks

################ Data Loader ####################
###############################################
def get_data_loader(batch_size, inputs, masks, labels):
	data = TensorDataset(inputs, masks, labels)
	sampler = RandomSampler(data)
	dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
	return data, sampler, dataloader



################ Transformer Model ####################
###############################################
def get_transformer_model(modelname):
	if modelname == "bert-base-uncased":
		# Later have to return model also
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
		# Load BertForSequenceClassification, the pretrained BERT model with a single 
		# linear classification layer on top. 
		model = BertForSequenceClassification.from_pretrained(
		    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
		    num_labels = 3, # The number of output labels--2 for binary classification.
		                    # You can increase this for multi-class tasks.   
		    output_attentions = False, # Whether the model returns attentions weights.
		    output_hidden_states = False, # Whether the model returns all hidden-states.
		)

		# Tell pytorch to run this model on the GPU.
		model.cuda()

	return tokenizer, model

################ Optimizer Scheduler ####################
###############################################
def get_optimizer_scheduler(name, model, train_dataloader_len, epochs):
	if name == "Adam":
		optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
		total_steps = train_dataloader_len * epochs

		# Create the learning rate scheduler.
		scheduler = get_linear_schedule_with_warmup(optimizer, 
		                                            num_warmup_steps = 0, # Default value in run_glue.py
		                                            num_training_steps = total_steps)
	return optimizer, scheduler


################ Flat Accuracy Calculation ####################
###############################################################

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

################ Time elapsed ####################
###############################################################

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


################ Main TRAINING CODE ###########################
###############################################################

def train(epochs, model, train_dataloader, validation_dataloader, optimizer, scheduler):
	print("In the Training Stage")
	# Set the seed value all over the place to make this reproducible.
	seed_val = 42

	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)

	# Store the average loss after each epoch so we can plot them.
	loss_values = []

	# For each epoch...
	for epoch_i in range(0, epochs):
	    
	    # ========================================
	    #               Training
	    # ========================================
	    
	    # Perform one full pass over the training set.

	    print("")
	    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
	    print('Training...')

	    # Measure how long the training epoch takes.
	    t0 = time.time()

	    # Reset the total loss for this epoch.
	    total_loss = 0

	    # Put the model into training mode. Don't be mislead--the call to 
	    # `train` just changes the *mode*, it doesn't *perform* the training.
	    # `dropout` and `batchnorm` layers behave differently during training
	    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
	    model.train()

	    # For each batch of training data...
	    for step, batch in enumerate(train_dataloader):

	        # Progress update every 40 batches.
	        if step % 40 == 0 and not step == 0:
	            # Calculate elapsed time in minutes.
	            elapsed = format_time(time.time() - t0)
	            
	            # Report progress.
	            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

	        # Unpack this training batch from our dataloader. 
	        #
	        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
	        # `to` method.
	        #
	        # `batch` contains three pytorch tensors:
	        #   [0]: input ids 
	        #   [1]: attention masks
	        #   [2]: labels 
	        b_input_ids = batch[0].to(device)
	        b_input_mask = batch[1].to(device)
	        b_labels = batch[2].to(device)

	        # Always clear any previously calculated gradients before performing a
	        # backward pass. PyTorch doesn't do this automatically because 
	        # accumulating the gradients is "convenient while training RNNs". 
	        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
	        model.zero_grad()        

	        # Perform a forward pass (evaluate the model on this training batch).
	        # This will return the loss (rather than the model output) because we
	        # have provided the `labels`.
	        # The documentation for this `model` function is here: 
	        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
	        outputs = model(b_input_ids, 
	                    token_type_ids=None, 
	                    attention_mask=b_input_mask, 
	                    labels=b_labels)
	        
	        # The call to `model` always returns a tuple, so we need to pull the 
	        # loss value out of the tuple.
	        loss = outputs[0]

	        # Accumulate the training loss over all of the batches so that we can
	        # calculate the average loss at the end. `loss` is a Tensor containing a
	        # single value; the `.item()` function just returns the Python value 
	        # from the tensor.
	        total_loss += loss.item()

	        # Perform a backward pass to calculate the gradients.
	        loss.backward()

	        # Clip the norm of the gradients to 1.0.
	        # This is to help prevent the "exploding gradients" problem.
	        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

	        # Update parameters and take a step using the computed gradient.
	        # The optimizer dictates the "update rule"--how the parameters are
	        # modified based on their gradients, the learning rate, etc.
	        optimizer.step()

	        # Update the learning rate.
	        scheduler.step()

	    # Calculate the average loss over the training data.
	    avg_train_loss = total_loss / len(train_dataloader)            
	    
	    # Store the loss value for plotting the learning curve.
	    loss_values.append(avg_train_loss)

	    print("")
	    print("  Average training loss: {0:.2f}".format(avg_train_loss))
	    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
	        
	    # ========================================
	    #               Validation
	    # ========================================
	    # After the completion of each training epoch, measure our performance on
	    # our validation set.

	    print("")
	    print("Running Validation...")

	    t0 = time.time()

	    # Put the model in evaluation mode--the dropout layers behave differently
	    # during evaluation.
	    model.eval()

	    # Tracking variables 
	    eval_loss, eval_accuracy = 0, 0
	    nb_eval_steps, nb_eval_examples = 0, 0

	    # Evaluate data for one epoch
	    for batch in validation_dataloader:
	        
	        # Add batch to GPU
	        batch = tuple(t.to(device) for t in batch)
	        
	        # Unpack the inputs from our dataloader
	        b_input_ids, b_input_mask, b_labels = batch
	        
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
	            outputs = model(b_input_ids, 
	                            token_type_ids=None, 
	                            attention_mask=b_input_mask)
	        
	        # Get the "logits" output by the model. The "logits" are the output
	        # values prior to applying an activation function like the softmax.
	        logits = outputs[0]

	        # Move logits and labels to CPU
	        logits = logits.detach().cpu().numpy()
	        label_ids = b_labels.to('cpu').numpy()
	        
	        # Calculate the accuracy for this batch of test sentences.
	        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
	        
	        # Accumulate the total accuracy.
	        eval_accuracy += tmp_eval_accuracy

	        # Track the number of batches
	        nb_eval_steps += 1

	    # Report the final accuracy for this validation run.
	    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
	    print("  Validation took: {:}".format(format_time(time.time() - t0)))

	print("")
	print("Training complete!")
	return model


################ Evaluation Code ##############################
###############################################################

def evaluate(prediction_dataloader, model):
	# Prediction on test set

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
	  b_input_ids, b_input_mask, b_labels = batch
	  
	  # Telling the model not to compute or store gradients, saving memory and 
	  # speeding up prediction
	  with torch.no_grad():
	      # Forward pass, calculate logit predictions
	      outputs = model(b_input_ids, token_type_ids=None, 
	                      attention_mask=b_input_mask)

	  logits = outputs[0]

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
	print('CS classification accuracy is')
	print(metrics.accuracy_score(true_labels, predictions)*100)
	print(classification_report(true_labels, predictions, target_names = ['neutral', 'against', 'for']))

################ Main Function ####################
###################################################

def main():
	print("Welcome to BERT Pytorch Sentence Classification Pipeline")

	# Loading the datasets
	print('Loading Covid-Stance dataset')
	train_df = pd.read_csv('datasets/cstance_train.csv')
	print(train_df.columns)

	# Test set
	test_df = pd.read_csv('datasets/cstance_test.csv')
	print(test_df.columns)

	(x1_train, x2_train, y_train), (x1_val, x2_val, y_val), (x1_test, x2_test, y_test) = get_data(train_df, test_df)

	# Geting the Transformer Tokenized Output
	MAX_LEN=100
	model_name = 'bert-base-uncased'
	batch_size = 32
	epochs = 4
	tokenizer, model = get_transformer_model(model_name)
	print("Successfully retrived tokenizer and the model!")
	train_inputs, train_masks = tokenize(x1_train, x2_train, tokenizer, MAX_LEN)
	val_inputs, val_masks = tokenize(x1_val, x2_val, tokenizer, MAX_LEN)
	test_inputs, test_masks = tokenize(x1_test, x2_test, tokenizer, MAX_LEN)

	# Converting the labels into torch tensors
	train_labels = torch.tensor(y_train, dtype=torch.long, device =device)
	val_labels = torch.tensor(y_val, dtype=torch.long, device =device)
	test_labels = torch.tensor(y_test, dtype=torch.long, device =device)

	# Printing the shape of these tensors
	print('Printing the shape of the final tensors')
	print('Train input', train_inputs.shape, 'Train Masks', train_masks.shape, 'Train Labels', train_labels.shape)
	print('Val input', val_inputs.shape, 'Val Masks', val_inputs.shape, 'Val Labels', val_inputs.shape)
	print('Test input', test_inputs.shape, 'Test Masks', test_inputs.shape, 'Test Labels', test_inputs.shape)
	# Getting the dataloaders
	train_data, train_sampler, train_dataloader = get_data_loader(batch_size, train_inputs, train_masks, train_labels)
	val_data, val_sampler, val_dataloader = get_data_loader(batch_size, val_inputs, val_masks, val_labels)
	test_data, test_sampler, test_dataloader = get_data_loader(batch_size, test_inputs, test_masks, test_labels)
	print("Successfull in data prepration!")

	# Getting optimzer and scheduler
	optimizer, scheduler = get_optimizer_scheduler("Adam", model, len(train_dataloader), epochs)
	print("Successfully loaded optimzer and scheduler")

	# Main Traning
	model = train(epochs, model, train_dataloader, val_dataloader, optimizer, scheduler)

	# Evaluation on Test set
	evaluate(test_dataloader, model)

if __name__ == "__main__":
    main()



