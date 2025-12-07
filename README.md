# Text-Summarization-Using-T5-Hugging-Face-Model

##  Project Overview  
This project implements a **dialogue summarization system** using the T5 model from Hugging Face.  
Given a long conversation (dialogue) as input, the system cleans and tokenizes the text, feeds it to a fine‑tuned T5 model, and outputs a concise summary in plain English.  

The goal is to enable quick summarization of dialogues — conversations, chats or meeting transcripts — into short, readable summaries.  

##  Data Preparation & Training  

1. Download a dialogue‑summary dataset (e.g. the Samsum Dataset from Kaggle).  
2. Load the data via pandas, clean text (remove extra spaces, line breaks, tags, etc.), drop missing values.  
3. (Optional) Downsample or shuffle data and reset the index — useful if working with large datasets or for quick tests.  
4. Use a preprocessing function to tokenize dialogues and summaries; assign tokenized summaries to `labels`.  
5. Initialize and fine‑tune a T5 model (e.g. `t5-small`) using a training framework (e.g. `Trainer` from Hugging Face).  
6. After training, save the model and tokenizer to a folder (e.g. `saved_t5_hugging_face_models/`).


  ##  Why This Matters  

- Dialogues and conversations can be long and hard to skim — summarization helps extract key points quickly.  
- Useful for chat logs, meeting transcripts, support conversations, user interviews, etc.  
- Offers automation: rather than manually summarizing, use a trained model to get instant summaries.  
