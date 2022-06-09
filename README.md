# Detecting_Deepfake_Videos_Based_on_Spatiotemporal_Attention_and_Convolutional_LSTM
the code of Detecting Deepfake Videos Based on Spatiotemporal Attention and Convolutional LSTM

# Paper Abstract

Face identification is in dilemma with the rapid development of face manipulation technology. Existing algorithms can no longer meet the fake face video detection requirements. One way to improve the effectiveness of detector is to make full use of intra and inter frame information. In this paper, a novel Xception-LSTM algorithm is proposed by using spatiotemporal attention mechanism and convolutional long short-term memory (ConvLSTM). The spatiotemporal attention mechanism, including spatial and temporal attention mechanism, is proposed to capture spatiotemporal correlations to remind Xception of the presence of temporal manipulated traces. Thereafter, the ConvLSTM is introduced to consider frame structure information while modeling temporal information. The experimental results on the widely used FaceForensics++ dataset demonstrate that the proposed algorithm performs better than the second best algorithm among compared eight algorithms with the 3.3% accuracy (ACC) and 0.019 area under curve (AUC). Moreover, the effectiveness of the spatiotemporal attention mechanism and ConvLSTM are illustrated in ablation experiments.

# HOW TO USE THE CODE
## Get Model
Download Model.py and Attention.py. The _`getMoel()`_ method in Model.py will return the model proposed in paper.
## Training
We recommend you write your own method for reading data and training.
If you want to use the code we provided in this repo to read data and train the network,
please follow these steps to prepare your dataset. 

1. Extract frames from videos
2. Save frames of the same video to the same folder
3. Generate a plain text file for each dataset you need. 
   Each line in the file includes the frame folder path and label, separated by spaces.

Once the data is ready, you need to rewrite the data paths in Train.py and run it for training.

## Testing
We recommend you write your own method for testing. 
The method we used to test model is wrote in NetTest.py.
You need to rewrite some paths before using it.
