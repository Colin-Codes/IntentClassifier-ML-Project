# IntentClassifier
Classifying the intent behind tech support emails in order to enable automated follow up, using K-Nearest Neighbours and Neural Network (RNN, LSTM and GRU) models

I have briefly documented the project below to enable a quick review of the project; for a more in-depth report, please refer to documents/16018361_FPR.docx (my dissertation document) and 16018361_Appendix.docx. 

# Processing the data
I started with only ~1000 records so sought to create balanced and augmented data sets early in the project. 

I used sampling methods firstly to split out a test set, but also to create a balanced dataset which contained an even balance of samples across each classification.

The EDA folder contains code by Jason Wei and Kai Zou that I have only in the breifest sense modified. This was instrumental in providing many more samples in the data set by expanding and randomly amending samples in the original data set. 

I was then able to compare original, balanced and augmented datasets in order to determine what value these techniques added.


# Training the models
Training models, particularly Neural Networks, is very computationally expensive - in order to complete the project I would need to write code that allowed me to set up experiments and let them run through the day and night without requiring oversight. This is what you can see in Main.py - the commented code represents completed experiments. 

I achieved this by creating three modules:
1. dataHelper.py - a data class that holds, processes and reports metadata as required
2. modelHelper.py - for building and training models
3. experimentHelper.py - an Experiments class that combines the above and adds reporting functionality to quickly produce graphs using matplotlib

I could then iterate over lists of parameters in order to search the space and find an optimal combination.


# Results
For a final comparison, please refer to results/global/graphs which compares optimal models from each model type for precision and recall. Interestingly, due to the training time of neural networks and the variety of avialable parameters I calculated that my original testing plan would have taken approxiamtely 5 years to complete. The final selection of parameters was made on a search of the space that took approximately 3 weeks and had to be based on a judgement of where value could best be added.

In this application, precision was to be more critical than recall, as the consequence of mis-classifying inputs could result in significant repercussions for the business.

The results of augmenting the data were mixed - it would be hard to argue that either the balanced or augmented datasets produced definitively better results and in fact the LSTM model produced significantly worse results when balanced OR augmented, which could indicate a high degree of bias.

The KNN model came out on top. Due to the simplicity of the model, it simply operated better under the pressure of having little data to work with. With much more data, I feel that the neural networks may have performed better and may have been easier to optimise - I have to wonder if a significant factor of 'improvement' I saw when varying parameters may simply have been due to statistical noise. Additionally, the neural networks all required a fixed sentence length which made them poorly suited to the application, where actual sentence length could vary dramatically (data/graphs/originalData.png). Perhaps ensemble models with varieties of sentence length could mitigate this somewhat.

I hope this brief overview has been helpful, if you have any questions please feel free to contact me via email at: colin.younge@gmail.com
