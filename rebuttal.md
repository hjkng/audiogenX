## Response to Reviewer 81aq
We sincerely appreciate your valuable feedback and time for reviewing this paper. For the concerns you bring up, we would like to address them as follows:

### Weakness 1: Scenario of AudioGenX.
Thank you for pointing this out. Explaining generated audio brings benefits in several cases: 1)  AudioGenX increases awareness of the impact of each input part, helping us ensure that the model focuses on the correct aspects of the text. 2) Explanations provide insight for users to plan the next prompting strategy when the previous audio generation is not satisfied. 3) When the user wants to edit the audio to amplify/suppress the impact of certain textual input, the importance may serve as the actionable information to decide how much to adjust the related weight involving the edition. Please refer to the attached pdf where the figure of the above scenario is described.

### Weakness 2: Addressing clarity in the abstract.
When AudioGen succeeds in describing audio in response to textual input, for example, the descriptive text as "Railroad crossing signal followed by a train passing and blowing horn," it is comparatively easy to understand outputs. However, it often fails to convert certain parts of textual input to audio. In that case, it is hard to know how much AudioGen considers the information quantitatively, leaving users in a state of uncertainty following possible questions like "How much a missing textual token is related to the current audio?" When the importance of the textual token is low, removing the token does not have much impact on the generated audio. Conversely, the token of high importance can significantly influence the audio, so this information helps users to make the next prompting strategy. AudioGenX aims to answer questions to provide an explanation quantifying the importance of each text token corresponding to the generated audio.

### Question 1: Comparison with zero-shot LLM.
While large language models like ChatGPT can identify significant sound-like words, this capability does not translate into a meaningful explanation for generated audio. AudioGen, which uses top-k or top-p sampling for generation, produces varied audio outputs with each trial. In this context, AudioGenX can detect and quantify distinctive features of each audio generated from the same textual input, providing differentiated explanations for each audio instance.

### Question 2: Explanation in case of negation and double negation.
We present explanations for cases involving negation and double negation, as detailed in the attached PDF. Using AudioGenX, we observe that the negative words "without" and "no without" have lower importance compared to "thunder" in the explanations. Interestingly, both "without" and "no without" result in generated audio that includes the sound of thunder. We hypothesize that this occurs because the training dataset lacks sufficient examples of negation and double negation. An examination of the AudioCaps dataset reveals a scarcity of such cases. Consequently, the generation model's limitations in handling negation are reflected in AudioGenX's explanations, which assign lower importance to these negation words.

----------------------------------------------------------------------------

## Response to Reviewer s1so
We sincerely appreciate your valuable feedback and time for reviewing this paper. For the concerns you bring up, we would like to address them as follows:

### Weakness 1: Addressing clarity and coherence.
We thank you for your thorough and valuable review. We agree that notations are quite numerous to represent the architecture of Audiogen originally where many components are composed. We have revised the related section of our paper more clearly.

### Weakness 2: Addressing the experimental setting
While explanation methods have been extensively explored in various domains, to our knowledge, no existing methods address explanations for text-to-audio generation models. Consequently, evaluation metrics for explainability in the audio generation domain do not exist yet. In other domains, fidelity is one of the prevalent metrics that measure the change of target class probability in supervised settings. However, In generation tasks, such ground truths and labels do not exist. When the same textual input is fed into the generation model, the output can vary due to top-p (nucleus) sampling. Thus, evaluating explanations is particularly challenging when ground truth or class labels are unavailable. Therefore, a fidelity or confusion matrix is inadequate in the absence of ground truths. Given the nature of the audio domain, KL divergence is widely employed in most audio generation research. This metric leverages the distribution of each class to represent the inherent meaning of audio. While traditional audio generation models measure KL divergence between generated audio and reference audio in datasets like AudioCaps, we measure the difference between generated audio and factual/counterfactual audio perturbed through optimized explanation masks. This allows us to observe the impact of explanation masks even in the absence of ground truths.

### Weakness 3: Applying our method in other seq2seq tasks.
We agree that there are no prior general methods regardless of domains and also our approach could expand in other seq2seq tasks, which is an interesting direction for our future research in explainability.

### Question 1: Non-transference from XAI for audio processing models to audio generation models.
The audio processing model has fundamental differences with audio generation models. First, the generation model has a different inference mode compared to supervised models, involving top-k and p sampling to predict the output. Since this is related to the next question, we explain the difference further in detail in question 2. Also, the type of input differs: audio processing models handle sequential audio input, dealing with a single modality, whereas audio generation models involve mainly textual input and in some cases conditional melody encompassing multi-modal data. Furthermore, since text-to-audio generation models are sequence-to-sequence models, requiring significant modification for the application of XAI methods.

### Question 2: Consideration of Top-k, p sampling in XAI.
AudioGenX must account for the nucleus sampling process involved in the audio generation of AudioGen, where the next audio token at each step is selected through a sampling process. This inference process differs from that of supervised models, where the final output is chosen based on the highest probability or logit. With sampling hyper-parameter k=200, the number of possible output candidates is 200. Thus, it is problematic to track the gradient of only one selected logit like XAI in supervised settings, because this method neglects the information of other audio tokens sharing similar implicit meanings. Therefore, we formulate factual and counterfactual explanations based on final latent embedding vectors before conducting the sampling process.

### Question 3: Audio generation models without cross-attention blocks.
Most current state-of-the-art audio generation models [1] still employ cross-attention blocks to fuse textual and auditory information. Beyond explainability, recent methods for editing [2] in generation settings also utilize the cross-attention layer to re-weight generation objects in the output, highlighting the importance of cross-attention blocks. Nevertheless, our perturbation methods can be applied to models trained with or without cross-attention blocks through minimal modifications, thanks to their adaptable scheme.

### Question 4-5: Addressing clarity of notations and definition of $h_{E}$.
Notations $a$ and $c$ indicate generated audio and given text, respectively, as shown in line 105. Specifically, audio $a$ can be any audio data generated by the decoder of the target model. We denote the attention block in the Transformer as $h_{E}$, as referenced in line 118.

### Question 6-7: Definition of the mask.
While the explanation mask is defined in lines 134 and 135, $m_{i}$ is an entity of explanation mask $M$.

### Question 8-11: Addressing clarity of the method.
We replace multiple MLPs with multiple linear layers and use the PReLU activation function. Unlike the standard softmax, the Gumbel-softmax function is employed to force values closer to 0 or 1 to observe a significant effect of perturbation. Since the explanation is the output from the explainer, the objective function is defined in Eq (7). As described in Algorithm 1 on page 6, our explainer is optimized using the loss function in Eq (7) for the explanation mask.

### Reference
[1] Masked Audio Generation using a Single Non-Autoregressive Transformer, ICLR, 2024. 

[2] Connecting Speech Encoder and Large Language Model for ASR, IEEE, 2024.

----------------------------------------------------------------------------

## Response to Reviewer rLwA
We sincerely appreciate your valuable feedback and time for reviewing this paper. For the concerns you bring up, we would like to address them as follows:

### Weakness 1: Addressing clarity.
We thank you for your valuable advice for the clarity of the manuscript. We have been revising the related section of our paper more clearly. In respect of perturbation and explanation mask, we answer it further in question 1.

### Weakness 2: Recommendation for additional datasets.


### Question 1: Optimizing explanation masks.
Algorithm 1 describes the procedure for obtaining explanations for text-audio pairs. Since audio is a sequential output that can be decomposed into audio tokens, we train the explainer to predict the mask at the audio token level. Thus, we obtain the explanation mask through the explainer optimized by explaining objective functions in Eq. (7). When perturbing the attention weight through the mask, we obtain factual and counterfactual results affecting factual maks $M$ and counterfactual mask $1-M$, respectively. Consequently, we aggregate the optimized mask in the audio token level into the audio level in section 4.3.

### Question 2: Visual analysis of the attention matrix vs explanation in AudioGenX.
Figure (3) in the attached image illustrates the effect of $M_{u, z_t}$ on the attention mechanism within AudioGen's first cross-attention block, displaying attention scores, mask values, and the product of both. Attention scores alone are insufficient for explanations because, post-softmax, they are multiplied by $V$, which significantly influences the outcome. Initially, keywords related to sounds such as "rain," "forest," and "thunder" have relatively low attention scores despite their factual importance. However, these tokens are assigned significantly higher mask values $M_{u, z_t}$, indicating their crucial role in sound generation. When multiplying the attention scores by the mask values, the important tokens maintain or increase their significance, while less important tokens see a reduction. As a result, the relative ranking of attention values ​​of important tokens increased and AudioGenX can make the final embeddings more similar to the reference audio embeddings. From a counterfactual perspective, reducing the significance of important tokens while increasing that of less important ones would lead to embeddings that diverge from the reference audio embeddings.


### Question 3: Hyper-parameter setting for coefficient β and γ.
Hyperparameter search for β and γ is conducted on the valid dataset which is described in line 240 in section 5.

### Question 4: Time complexity of AudioGenX and ATMAN.
The time complexity of ATMAN is $O(T_u)$, where $T_u$ represents the number of textual tokens. ATMAN extracts important tokens through single-token perturbation, requiring $T_u$ iterations. In comparison, AudioGenX has a time complexity of $O(T_u k)$, with $k$ denoting the number of epochs. Although both $O(T_u)$ and $O(T_u k)$ increase linearly with respect to $T_u$, making their time complexities equivalent in terms of growth rate, the actual difference in $k$ results in approximately five times the inference time for AudioGenX.

----------------------------------------------------------------------------

## Response to Reviewer 9Txk
We sincerely appreciate your valuable feedback and time for reviewing this paper. For the concerns you bring up, we would like to address them as follows.

### Weakness 1: Addressing readability.
Thank you for your suggestions. We realized that some sentences, like the one you quoted, were not rigorous enough. We have revised the manuscript accordingly to be more clear and precise.

### Weakness 2: Addressing questions for ML explanation methods.
We agree that the definition of faith explanation is crucial to meeting the criteria. While proper evaluating metrics are required, such ground truths and labels do not exist in generation tasks. When the same textual input is fed into the generation model, the output can vary due to top-p (nucleus) sampling. Thus, we evaluate the explanations upon the metrics, KL-divergence using the PaSST audio classifier, commonly applied in audio generation models but modified in the context of explainability. Besides evaluation metrics, we conduct the sanity check by generating explanations for random initialized networks that predict the audio tokens. The results are shown in Figure (3) of the attached PDF. Unlike the explanations generated based on the trained AudioGen, all models, including the baseline, produced results where the influence of each token was nearly indistinguishable. This indicates that AudioGen generates faithful explanations.


### Weakness 3: Benefits of the proposed explanations.
Thank you for pointing this out. While explainability serves different roles for each stakeholder, we demonstrate a case study leveraging explainability to analyze the pattern of AudioGen in Section 6.2, RQ 2. Besides debugging the model,  explaining generated audio brings benefits in several cases: 1)  AudioGenX increases awareness of the impact of each input part, helping us ensure that the model focuses on the correct aspects of the text. 2) Explanations provide insight for users to plan the next prompting strategy when the previous audio generation is not satisfied. 3) When the user wants to edit the audio to amplify/suppress the impact of certain textual input, the importance may serve as the actionable information to decide how much to adjust the related weight involving the edition. Please refer to the attached pdf where the figure of the above scenario is described.

### Weakness 4: Addressing typos.
We thank you for your thorough review. We revised the mentioned typo accordingly. 

### Question 1: Inquiry of the procedure.
Algorithm 1 describes the procedure for obtaining explanations for text-audio pairs. Since audio is a sequential output that can be decomposed into audio tokens, we train the explainer to predict the mask at the audio token level, denoted as $Explainer_{token}$. Line 259 refers to $Explainer_{token}$, which generates the explanation mask. The input to the explainer is the representation vector of the textual input token. Thus, dim = 1536 indicates the dimension of the representation vector of the textual input token, and dim = 512 refers to the hidden dimension of the hidden layer in the explainer. After reading your comment, we also realized some sentences like the one you quoted here are not appropriate. In Algorithm 1, the initialization should be placed outside the loop, we have revised line 4 in the algorithm. 


----------------------------------------------------------------------------

## Response to Reviewer yHGn
We sincerely appreciate your valuable feedback and time for reviewing this paper. For the concerns you bring up, we would like to address them as follows:

### Weakness 1: Addressing novelty.
While perturbation-based explanation methods, including factual and counterfactual approaches, have been extensively explored and proven effective in various domains, to our knowledge, no existing methods address explanations for text-to-audio generation models. Furthermore, no studies have demonstrated the efficacy of perturbation-based reasoning in audio generation. We conducted experiments using state-of-the-art explainability methods upon the metrics commonly applied in audio generation models but modified in the context of explainability. Our results indicate that AudioGenX significantly outperforms existing explanation methods originally from other domains in generating explanations. In conclusion, our contribution lies in pioneering transparency solutions in the audio domain and proposing faithful explanation methods for the first time.

### Weakness 2: Addressing aptness of evaluation metrics.
Evaluating explanations is particularly challenging when ground truth or class labels are unavailable. In generation tasks, such ground truths and labels do not exist. When the same textual input is fed into the generation model, the output can vary due to top-p (nucleus) sampling. Therefore, a confusion matrix is inadequate in the absence of ground truths. Given the nature of the audio domain, KL divergence is widely employed in most audio generation research. This metric leverages the distribution of each class to represent the inherent meaning of audio. While traditional audio generation models measure KL divergence between generated audio and reference audio in datasets like AudioCaps, we measure the difference between generated audio and factual/counterfactual audio perturbed through optimized explanation masks. This allows us to observe the impact of explanation masks even in the absence of ground truths.

### Question 1: Suggestion for another training method to improve the audio generation quality.
Thanks for your kind suggestions. Employing our approach to enhance the quality of audio generation is an interesting topic but the mentioned methods may bring several issues. First, evaluation techniques involving audio classification models require audio input, necessitating the generation of the full sequence of audio at every epoch when optimizing the explainer. This makes parallel generation impractical and significantly increases computational time. Second, the redundant process of converting discrete audio tokens to waveforms using Encodec is mandatory for employing an audio classifier, further increasing computational costs. Third, obtaining gradients for optimizing the explainer is unreliable because the top-p or top-k sampling parts must be modified to differential sampling. To address these challenges, we reformulate the problem by explaining sequential output at the token level to generate explanations efficiently. Specifically, we multiply the importance weights with cross-attention weights to generate factual and counterfactual audio, allowing us to observe the impact of respective explanation masks on the generated audio. In conclusion, we hold our proposed framework in sequential output not only for explainability but also for other purposes.
