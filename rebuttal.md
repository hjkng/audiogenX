## Response to Reviewer 81aq
We sincerely appreciate your valuable feedback and time for reviewing this paper. For the concerns you bring up, we would like to address them as follows.

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
We sincerely appreciate your valuable feedback and time for reviewing this paper. For the concerns you bring up, we would like to address them as follows.

### Weakness 1: Addressing clarity and coherence.
We thank you for your thorough and valuable review. We agree that notations are quite numerous to represent the architecture of Audiogen originally where many components are composed. We have revised the related section of our paper more clearly.

### Weakness 2: Addressing the experimental setting
While explanation methods have been extensively explored in various domains, to our knowledge, no existing methods address explanations for text-to-audio generation models. Consequently, evaluation metrics for explainability in the audio generation domain do not exist yet. In other domains, fidelity is one of the prevalent metrics that measure the change of target class probability in supervised settings. However, In generation tasks, such ground truths and labels do not exist. When the same textual input is fed into the generation model, the output can vary due to top-p (nucleus) sampling. Thus, evaluating explanations is particularly challenging when ground truth or class labels are unavailable. Therefore, a fidelity or confusion matrix is inadequate in the absence of ground truths. Given the nature of the audio domain, KL divergence is widely employed in most audio generation research. This metric leverages the distribution of each class to represent the inherent meaning of audio. While traditional audio generation models measure KL divergence between generated audio and reference audio in datasets like AudioCaps, we measure the difference between generated audio and factual/counterfactual audio perturbed through optimized explanation masks. This allows us to observe the impact of explanation masks even in the absence of ground truths.

### Weakness 3: Applying our method in other seq2seq tasks.
We agree that there are no prior general methods regardless of domains and also our approach could expand in other seq2seq tasks, which is an interesting direction for our future research in explainability.

### Question 1: Non-transference from XAI for audio processing models to audio generation models.
The audio processing model has fundamental differences with audio generation models. First, the type of input differs: audio processing models handle sequential audio input, dealing with a single modality, whereas audio generation models involve textual input, encompassing multi-modal data. Furthermore, since text-to-audio generation models are sequence-to-sequence models, requiring significant modification for the application of XAI methods.


### Question 2: Consideration of Top-k, p sampling in XAI

AudioGenX must account for the nucleus sampling process involved in AudioGen, where the next audio token at each step is selected through a sampling process. This inference process differs from that of supervised models, where the final output is chosen based on the highest probability or logit. With sampling hyper-parameter k=200, the number of possible output candidates is 200. Thus, it is problematic to track the gradient of only one selected logitlike XAI in supervised settings, because this method neglects the information of other audio tokens sharing similar implicit meanings. Therefore, we formulate factual and counterfactual explanations based on final latent embedding vectors before conducting the sampling process.



### Question 3: XAI for other type of audio generation models

### Question 4: Addressing clarity of notations

### Question 5: Definition of $h_{E}$

### Question 6-7: Definition of the mask

### Question 8-11: 



----------------------------------------------------------------------------

## Response to Reviewer rLwA
We sincerely appreciate your valuable feedback and time for reviewing this paper. For the concerns you bring up, we would like to address them as follows.

### Weakness 1: Addressing clarity.

### Weakness 2: Addressing clarity.


### Question 1: Optimizing explanation masks.

### Question 2: Visual analysis of the attention matrix vs explanation in AudioGenX.

### Question 3: Hyper-parameter setting for coefficient β and γ

### Question 4: Time complexity of AudioGenX and ATMAN


----------------------------------------------------------------------------

## Response to Reviewer 9Txk
We sincerely appreciate your valuable feedback and time for reviewing this paper. For the concerns you bring up, we would like to address them as follows.

### Weakness 1: Addressing readability.

### Weakness 2: Addressing lack of readability.

### Weakness 3: Benefits of the proposed explanations.

### Weakness 4: Addressing typos.

### Question 1: Inquiry of overall procedure.



----------------------------------------------------------------------------

## Response to Reviewer yHGn
We sincerely appreciate your valuable feedback and time for reviewing this paper. For the concerns you bring up, we would like to address them as follows.

### Weakness 1: Addressing novelty.
While perturbation-based explanation methods, including factual and counterfactual approaches, have been extensively explored and proven effective in various domains, to our knowledge, no existing methods address explanations for text-to-audio generation models. Furthermore, no studies have demonstrated the efficacy of perturbation-based reasoning in audio generation. We conducted experiments using state-of-the-art explainability methods upon the metrics commonly applied in audio generation models but modified in the context of explainability. Our results indicate that AudioGenX significantly outperforms existing explanation methods originally from other domains in generating explanations. In conclusion, our contribution lies in pioneering transparency solutions in the audio domain and proposing faithful explanation methods for the first time.

### Weakness 2: Addressing aptness of evaluation metrics.
Evaluating explanations is particularly challenging when ground truth or class labels are unavailable. In generation tasks, such ground truths and labels do not exist. When the same textual input is fed into the generation model, the output can vary due to top-p (nucleus) sampling. Therefore, a confusion matrix is inadequate in the absence of ground truths. Given the nature of the audio domain, KL divergence is widely employed in most audio generation research. This metric leverages the distribution of each class to represent the inherent meaning of audio. While traditional audio generation models measure KL divergence between generated audio and reference audio in datasets like AudioCaps, we measure the difference between generated audio and factual/counterfactual audio perturbed through optimized explanation masks. This allows us to observe the impact of explanation masks even in the absence of ground truths.

### Question 1: Suggestion for another training method to improve the audio generation quality.
Unfortunately, employing the mentioned methods is not feasible for several reasons. First, evaluation techniques involving audio classification models require audio input, necessitating the generation of the full sequence of audio at every epoch when optimizing the explainer. This makes parallel generation impractical and significantly increases computational time. Second, the redundant process of converting discrete audio tokens to waveforms using Encodec is mandatory for employing an audio classifier, further increasing computational costs. Third, obtaining gradients for optimizing the explainer is unreliable because the top-p or top-k sampling parts must be modified to differential sampling. To address these challenges, we reformulate the problem by explaining sequential output at the token level to generate explanations efficiently. Specifically, we multiply the importance weights with cross-attention weights to generate factual and counterfactual audio, allowing us to observe the impact of respective explanation masks on the generated audio. Notably, we do not utilize reference audio (which typically means the original sound corresponding to specific descriptive text in the AudioCaps dataset) for evaluation but the generated audio.
