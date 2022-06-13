# A-Brazillian-Portugese-BERT-Based-ML-System-For-Solution-Retrieval

## INTRODUCTION
This ML system was designed using a pretrained Brazillian BERT model at its core to retrieve the solutions to similar customer complaints (in text format) when given a new customer complaint.
The BERT model was downloaded from the huggingface transformer library. It uses the Numpy library for batch processing.

## INPUT
A customer complaint in brazillian portugese

## OUTPUT
Retrieve ranked solutions of the most similar complaints (to index customer complaint).

# METHOD
When given text (customer complaint), it uses the pretrained BERT model to get an embedded vector representation (using it's last layer) of this text and all complaints in the database
(faster to have all the complaints in database represented in vectors and stored somewhere). A cosine similarity is then used to calculate similarities between the index complaint and 
those in the database, ranking them in descending order.

The corresponding solutions to the ranked complaints are then retrieved from the database and the best returned to the index customer.

For more info on the code, architecture and explanations, read the .ipynb file attached to this repository.
