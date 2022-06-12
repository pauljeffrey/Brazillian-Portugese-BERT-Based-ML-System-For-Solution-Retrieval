import os
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining , AutoConfig, AutoModel
from time import time
import huggingface_hub as hb
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import pandas as pd
import torch
import warnings


FILE_PATH = os.path.absapth(os.curdir)
model_path = os.path.join(FILE_PATH,'bert-base-portuguese-cased')
# The below code forces the model to look offline (in the local directory) for its weight and dependencies.
TRANSFORMERS_OFFLINE=1

def download_BERT(download_dir):
    hb.snapshot_download('neuralmind/bert-base-portuguese-cased',cache_dir= download_dir)
    for f in os.listdir():
        if f.startswith('neuralmind__bert-base'):
            os.rename(f,'bert-base-portuguese-cased')
    return


def load_vectorizer(model_path=model_path):
  ''' This function loads the BERT model to memory for vectorization'''
  # Load the BERT model:
  model = AutoModel.from_pretrained( model_path, local_files_only = True, output_attentions=True)
  # Load the tokenizer:
  tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

  # Return the model and tokenizer:
  return model, tokenizer


def vectorize_all_defects(all_defects, model,tokenizer,max_length=128, batch_size=1000):
  ''' This function vectorizes 'all_defects' in batches (using the provided model and tokenizer arguments) 
      and concatenates all the resulting embeddings into one tensor (array) container
  '''
  # First tokenize and vectorize the first batch from 'all_defects'
  # Tokenize first batch:
  tokens = tokenizer.batch_encode_plus(all_defects,max_length= max_length, padding='max_length',truncation=True,return_tensors='pt')
  attention_mask = tokens['attention_mask']

  # Vectorize first batch and store result in 'embeddings' variable.
  with torch.no_grad():
    outs = model(**tokens)
    # The vector embeddings are stored in the last_hidden_state of the model so we retrieve it.
    embeddings = outs.last_hidden_state 

  # Make attention mask have exactly the same size and shape as the vectorized embeddings:
  # This is done because not all defects have similar sequence length and so they were all padded to the same size.
  # Therefore, attention mask here is used to ignore all the paddings.
  attention_mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
  embeddings = embeddings * attention_mask

  # summed_embeddings = torch.sum(embeddings, 1)
  # summed_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
  # The embeddings are done for each words in each defect description. Therefore, calculating the mean of each word embeddings per defect description
  # gives a standard embedding (mean) of each defect description.
  mean_pooled_embeddings = torch.sum(embeddings, 1) / torch.clamp(attention_mask.sum(1), min=1e-9)

  # Now, apply the same process to other batches and continue concatenating to the 'embeddings' variable till there are no defects left:
  for i in range(batch_size, len(all_defects), batch_size ):
  # Tokenize_batch
    tokens = tokenizer.batch_encode_plus(sentences,max_length=max_length, padding='max_length',truncation=True,return_tensors='pt')
    attention_mask = tokens['attention_mask']

    with torch.no_grad():
      outs = model(**tokens)
      embeddings = outs.last_hidden_state
    
    attention_mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    embeddings = embeddings * attention_mask
    # Calculate the mean of all word embeddings in each defect description:
    embeddings = torch.sum(embeddings, 1) / torch.clamp(attention_mask.sum(1), min=1e-9)
    # Concatenate this calculated mean_embeddings to the mean_pooled_embeddings:
    mean_pooled_embeddings = torch.cat((mean_pooled_embeddings, embeddings),0)

  # Return all embeddings:
  return mean_pooled_embeddings.detach().numpy() 


def get_success_rate(solution_id, success_status):
  ''' 
    Function calculates the success_rate (probability) of all solutions retrieved from the database. A more sophisticated function version.
    returns : a list of float values that represent the success probability for each solution.
  '''
  # Get the unique ids:
  # Create a dictionary and use each id in unique id as a key:
  success_rate_dict = { id : 0 for id in set(solution_id)}
  # Create a success rate list to store the probability of success:
  success_rate = []

  #index = np.arange(len(solution_id))

  # Expand dimensions of the solution_id and success_status list objects so we can concatenate them for faster processing:
  solution_id = np.expand_dims(solution_id, axis=1)
  success_status = np.expand_dims(success_status, axis=1)

  # Concatenate them into a 2-dimensional numpy array:
  solution_matrix = np.concatenate([solution_id, success_status],axis=-1)

  # For each id, filter out all occurences in the list and calculate the probability of success (mean):
  # This requires that the success status be represented as '1' for successful and '0' for unsuccessful.
  # Success probability is calculated here using the second column (to calculate the mean after filtering the occurences of each id) in the 2-dimensional array:
  for id in success_rate_dict.keys():
    success_rate_dict[id] = solution_matrix[solution_matrix[:,0] == id][:,1].astype('int').mean()

  # Assign the appropriate success probability to the solution id using the success_rate_dict dictionary.
  for id in solution_id[:,1]:
    success_rate.append(success_rate_dict[id])

  # Return success_rate
  return success_rate


def get_prob_success(solution_id, solution_counter):
  ''' 
    Function calculates the success_rate (probability) of all solutions retrieved from the database. This is the function used by the algorithm
    written below this function definition.
    returns : a list of float values that represent the success probability for each solution.
  '''
  # Convert solution_counter to a numpy array:
  solution_counter = np.array(solution_counter)
  # Calculate the total counts of all solutions:
  total_count = solution_counter.sum()
  # Calculate and round up success_percentage to 1 decimal place:
  success_percentage = np.round((solution_counter / total_count * 100), 1)
  return success_percentage
 

def query_database(top_defects_ids):
  ''' This function queries the database given the ids of the most similar defect descriptions and a condition (whether to return L1 solutions). 
      It has been specifically left to be created by your team depending on the database management software that is being used.
      It should return solution_ids , solution_level , solution_counter. Each of these columns should be a list.
      Function should return None, None, None for these columns if no solution was found in the database.'''
  pass

# Define a sort_score function to sort the similarity scores in descending order.
def sort_score(scores):
  '''
    The function sorts the similarity score in descending order and returns it.
  '''
  result = []
  for each in scores:
    result.append(each)
  # Sort the list according to the similarity scores:
  result.sort(key=lambda x: x[1],reverse=True)
  return np.array(result)

def get_solution(work_order, all_defect_ids, all_defect_descriptions, tokenizer,
                 bert_model, level='customer', threshold_score=0.8, batch_size= 1000, top_k=-1, return_type='json'):
  '''This function using deep learning, retrieves and returns the solutions and respective success_rates of the most similar workorder to the work order
     or description provided as one of its input. It returns an empty dictionary if no viable solution was found in the database.
     Args:
      work_order: string, a description of the defect created by a customer/ techinician
      all_defect_ids: Ids of all defects stored in the database
      all_defect_descriptions: Description of all defects stored in the database.
      tokenizer: BERT model tokenizer
      bert_model: BERT model used as vectorizer by this function.
      level: 'customer' or 'technician'. It describes the person who created the work_order
      threshold_score: float, default == 0.8. It determines the similarity threshold score to use when retrieving the top_K similar defects.
      batch_size : integer , default == 1000, the number of defect descriptions to process/vectorize at once using multiprocessing.
      top_k: integer, default == -1. The number of defects to consider after similarity score computation.
      return_type: string, default == 'json'. It determines whether the function should return a dictionary or json object.
  '''
  # Convert list of defect_ids to a numpy array:
  all_defect_ids = np.array(all_defect_ids)

  ## Pass customer work order/tech_defect_description and all_defect_descriptions to vectorizer: BERT model 
  # Add the 'work order' to the 'all_defect_descriptions' for computational convenience:
  all_defect_descriptions.insert(0, work_order)

  # Compute vectorized embeddings:
  vectorized_embeddings  = vectorize_all_defects(all_defect_descriptions, bert_model, tokenizer)
  
  # Calculate the cosine similarity between customer work order and all other descriptions extracted from the database:
  similarity_scores = cosine_similarity([vectorized_embeddings[0]], vectorized_embeddings[1:])

  # Concatenate defect_ids to vectorized_defect_descriptions:
  # First expand the dimensions of both 'all_defect_ids' and 'similarity_scores' to a 2-dimensional array:
  all_defects_ids = np.expand_dims(all_defect_ids, 1)
  n_samples  = similarity_scores.shape[-1]
  # Reshape similarity scores:
  similarity_scores = similarity_scores.reshape(n_samples, 1)

  # Concatenate both into a single array:
  similarity_scores_id = np.concatenate((all_defects_ids, similarity_scores), axis= -1)

  # Filter the top k defects greater than threshold score:
  top_defects = similarity_scores_id[ similarity_scores_id[:,1] > threshold_score]

  # Sort similarity scores:
  top_defects  = sort_score(top_defects)

  # Extract only the top_k ids for this descriptions with high similarity score:
  top_defect_ids  = top_defects[: top_k,0]

  # Query database with this ids here and return solution_ids ,  solution_levels, solution's success percentage:
  # If its at customer level return all types of solutions:
  # query function is to be defined based on the database management system used.
  # query function should return (None, None, None ,None) if no solutions were retrieved.

  if  level == 'customer':
    solution_ids , solution_level , solution_counter = query_database(topic_defect_ids)
  else: # Else return only level 2 and 3 solutions:
    solution_ids , solution_level , solution_counter = query_database(topic_defect_ids)
  
  # If there are no solutions ( we know this by checking if the variable solution_ids == None (is empty))
  if solution_ids == None:
    # Create an empty dictionary and return it as json object.
    result = {}
  else:
    # Calculate the success percentage:
    success_percentage = get_prob_success(solution_ids, solution_counter)

    # Group by solution level:
    # First, define a dictionary for each level solution.
    # If its customer order, create a level 1 dictionary alongside:
    if type == 'customer':
      level_1 = {}

    level_2 = {}
    level_3 = {}

    # Group by solution level into the appropriate level dictionaries as created above:
    # Data structure used for each level dictionary: a dict of {solution_ids: success_percentage}
    for index, s_level in enumerate(solution_level):
      if s_level == 'L1':
        # if solution level for this index solution is level 1, then store solution_id and success percentage as key , value pairs in the level_1 dictionary.
        level_1[solution_ids[index]] = success_percentage[index]
      elif solution_levels == 'L2':
        # Else if level 2 , store in the level 2 dictionary.
        level_2[solution_ids[index]] = success_percentage[index]
      else:
        # Else store in the level 3 dictionary.
        level_3[solution_ids[index]] = success_percentage[index]

    # Result contains :  order_id , order_type (whether technician or customer), work order , +/- level 1 solutions, level 2 solutions, level 3 solutions.
    # If a customer solution (L1 solution) is enabled, return level 1 solution alongside. If not, return level 2 and 3 solution only.
    if type == 'customer':
      result = {'Level 1': level_1, 'Level 2': level_2, 'Level_3': level_3}
    else:
      result = {'Level 2': level_2, 'Level_3': level_3}

  # Return result as dictionary or json object:
  if return_type == 'json':
    result = json.dumps(result)
  
  return  result