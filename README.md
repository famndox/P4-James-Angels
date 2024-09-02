<img src="Resources/Images/angels2.png"/>
Consider the irony of using machine learning to interpret humanity’s Mythopoeic Corpus, our many cultural tales, to reimagine a common human story.  

## Building a Text Generator Model

#### Training Materials
For the scale of this project, we're sampling folktales and brief myths; nothing of a narrative length. The idea is large, and compiling takes time, so we’ve consolidated a few datasets together which appear below in order of volume - or can be found in /Resources/Datasets:

*  Small collection of creation myths
	1. Courtesy of: Freely hosted, take a look using a <a href='https://railsback.org/CS/CSIndex.html'>Web View</a> or see the exact <a href='https://superpowerexperts.com/wp-content/uploads/2019/06/Creation-Stories-from-around-the-World.pdf'>PDF Link Used</a>
	2. Transformation: manual
	3. Number of Stories: 20
	4. Char Count: 94504 
	5. Format: .csv and .txt 
	6. Native American, Norse, Japanese, Babylonian, Hebrew, Maori…
*  "Medium" collection of Brothers Grimm
	1. Courtesy of: <a href='https://www.cs.cmu.edu/'>Carnegie Mellon University</a> via <a href='https://hub.packtpub.com/tensorflow-lstm-that-writes-stories-tutorial/'>>packt<hub</a>
	2. Transformation: see [/scripts/Artifacts](https://nbviewer.jupyter.org/github/famndox/P4-James-Angels/blob/main/Scripts/Artifacts.ipynb) for rip and zip scripts
	3. Number of Stories: 100
	4. Char Count: 914271
	5. Format: .txt
	6. Cultures: Germanic
*  "Large" collection of folklore
	1. Courtesy of: <a href='https://www.kaggle.com/datasets/andrzejpanczenko/folk-tales-dataset'>Andrzej Panczenko on Kaggle</a>
	2. Transformation: see [/scripts/Artifacts](https://nbviewer.jupyter.org/github/famndox/P4-James-Angels/blob/main/Scripts/Artifacts.ipynb) for data manipulation scripts
	3. Number of Stories: 2838
	4. Char Count: 25797400
	5. Format: .csv, .txt, and .xlsx (for Tableau)
	6. Cultures: 65 referenced from Albanian to Zimbabwean

#### Model Progression

Our recent utility of TensorFlow led us to a tutorial of an RNN model which began our journey. Several attempts were made using our smallest datasets to effectively use TF and Keras to train on varying RNN loops, and an LSTM (not appearing in this summary), before increasing the training set size. This is our journey in an attempt to create a "next-char-precition" model:

* We began with [/scripts/scratch](https://nbviewer.jupyter.org/github/famndox/P4-James-Angels/blob/main/Scripts/scratch.ipynb), which was an attempt at this <a href='https://www.tensorflow.org/text/tutorials/text_generation'>TensorFlow Tutorial</a>, using our smallest dataset. As you can see, the accuracy of 100 epochs was solidly at 98%, but our results were nonsensical.   
	* <img src="Resources/Images/scratch_acc.png"/>
 	* <img src="Resources/Images/scratch_one_step.png"/>
* [/scripts/scratch_folk](https://nbviewer.jupyter.org/github/famndox/P4-James-Angels/blob/main/Scripts/scratch_folk.ipynb) was an attempt to modify the RNN training loop and use the largest dataset, which yielded the same result.
	* <img src="Resources/Images/tbd.png"/>
* In [/scripts/scratching](https://nbviewer.jupyter.org/github/famndox/P4-James-Angels/blob/main/Scripts/scratching.ipynb), we’re fitting our Grimm dataset into <a href='https://gist.github.com/karpathy/d4dee566867f8291f086'>Andrej Karpathy's Minimal character-level Vanilla RNN</a> model, which was what the first tutorial was ultimately based on.
	* <img src="Resources/Images/sing_best.png"/>
* [/scripts/scratching_folk](https://nbviewer.jupyter.org/github/famndox/P4-James-Angels/blob/main/Scripts/scratching_folk.ipynb) was the same attempt using the more substantial training set. The generation results of this model are still nonsense, but at least some of our keywords are showing up consistently.
	* <img src="Resources/Images/sing_f_850k.png"/>
	* <img src="Resources/Images/sing_f_900k.png"/>

The best logged result came from *our* SCRATCHING model using the Grimm dataset (towards the m’th iterations). Not bad for a little next character predictor, but it looks like we're going to need to lean on something pre-trained. 

Notes on notebooks: 
- *the known good tutorial based models used in this narrative are those that can be found in /scripts*
- *original, but not necessarily current, and separate notebooks documenting our 'from-scratch' models' beginnings can be found in /scripts/xPlor*
- *the same collection of 'pre-trained' models used in the next section can be found in .virtuatl_documents/just_exploring*
- *eg. the pt_bert_bot notebook referenced below was derived from Project_4_BERT.ipynb, sans the learning rate config*
	
  
## Leveraging BERT and GPT-2 for Emotional Analysis and Text Generation

 
#### Scope
We attempted to fine tune the following pre-trained models to both provide a sentiment analysis and generate a story from our larger collection of folk stories. We were able to complete the analysis and 12 epochs of training, in about 20 hours, running [/scripts/pt_bert_bot](https://nbviewer.jupyter.org/github/famndox/P4-James-Angels/blob/main/Scripts/pt_bert_bot.ipynb)
 on a local cpu (jupyterlab + tensorflow).  
* Hugging Face Models:
	* Sentiment Analysis: Utilize <a href='https://huggingface.co/docs/transformers/en/model_doc/bert'>BERT</a> to analyze the emotional content of a collection of folk tales.
	* Story Generation: Leverage <a href='https://huggingface.co/openai-community/gpt2'>GPT-2</a> to generate new stories based on the dominant sentiment identified in the input text.
* Challenges: Address the limitations of working within free computational resources while attempting to produce high-quality, machine-generated stories.

#### Project Workflow
-	1. Sentiment Analysis with <a href='https://huggingface.co/docs/transformers/en/model_doc/bert'>BERT</a>:
We began by feeding a dataset of folk tales into BERT, a model known for its deep understanding of language nuances. 		
The model was tasked with categorizing each story based on its emotional tone into one of five categories: "very negative," "negative," "neutral," "positive," and "very positive."
The analysis results allowed us to identify the most prevalent sentiment in the dataset.
 	2. Story Generation with <a href='https://huggingface.co/openai-community/gpt2'>GPT-2</a>:
Using the dominant sentiment as a foundation, we crafted a prompt to feed into GPT-2, a model designed for generating text.
The GPT-2 tokenizer converted the prompt into tokens that the model could process.
Parameters such as beam search and no_repeat_ngram_size were fine-tuned to optimize the output and reduce repetition, aiming to generate a coherent and unique story.
 	3. Challenges Encountered:
Due to limitations in computational resources provided by free platforms, we faced significant challenges in achieving the desired output quality.
The stories generated by GPT-2 were often disjointed, repetitive, and lacked the coherence we aimed for, highlighting the need for more powerful hardware to train such models effectively.

#### Key Learnings
Understanding BERT and GPT-2: This project provided valuable insights into the strengths and limitations of using BERT for sentiment analysis and GPT-2 for text generation.

## Folk Tales Sentiment Analysis

#### Scope
We aimed to explore the emotional tone and subjective nature of folk tales from various cultures around the world. By utilizing natural language processing (NLP) techniques, this part of the project analyzes the sentiment (polarity) and subjectivity of each tale. This analysis offers insights into the cultural storytelling traditions and the emotional impact of these stories, helping to uncover the common themes and differences across different nations' folk narratives.

- File Folder: Sentiment Files
- Resource Used: <a href='https://www.kaggle.com/datasets/andrzejpanczenko/folk-tales-dataset'>Andrzej Panczenko on Kaggle</a>
- Libraries imported: langdetect, wordcloud, seaborn, textblob, pycountry, pandas and matplotlib.pyplot

#### Findings
-	1. Nations and Stories Classification
The notebook classifies folk tales by their nation of origin. Classification allows for cultural comparisons and understanding the diversity of sentiment across different folk traditions.
	2. The sentiment analysis outputs provide the polarity and subjectivity of various folk tales. Polarity ranges from -1 (negative sentiment) to 1 (positive sentiment), while subjectivity indicates the degree to which the text is subjective or objective.
For instance, the tale "Momotaro" from Japan has a positive polarity of approximately 0.24 and a moderate subjectivity of 0.45. This suggests that the story is somewhat positive and moderately subjective.
	3. Data Table Summaries: 
The data tables in the notebook contain columns such as source, nation, title, text, polarity, and subjectivity. These tables summarize the key attributes of the folk tales analyzed, including their emotional tone and the degree of personal opinion or factual content.
	4. Polarity and Subjectivity Trends:
The summary outputs highlight patterns in polarity and subjectivity across different nations. This can be significant for understanding the emotional and narrative styles prevalent in folk tales from different cultures.
	5. Overall Significance:
The analysis in this notebook provides insights into the emotional and subjective nature of folk tales from various cultures. By classifying and analyzing these tales, the project sheds light on cultural storytelling traditions and their emotional impact.

## Tableau


## Additional References
<a href=''>Kaggle</a>
<a href=''>Kaggle</a>
<a href=''>Kaggle</a>
<a href=''>Kaggle</a>
<a href=''>Kaggle</a>
<a href=''>Kaggle</a>

---


