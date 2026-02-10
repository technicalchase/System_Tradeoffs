# Team 3: Intelligent Tutoring Systems R Us

## Dataset description.
### Overview.
This project uses a small synthetic image dataset created for a three-way classification task. The dataset simulates student attention states in a classroom or study setting. 

### What does the dataset represent (use case)?
This setup provides a simplified but relevant proxy for real-world problems in **AI in Education** (attention monitoring, adaptive feedback) while remaining lightweight enough for course assignments. We use this as a proof of concept to study **membership inference attack (MIA)**.

### How big is it (#samples, label)?
- We use 40 student images captured under three gaze categories. This reflects a sample size of 120 images. 
- For the MIA evalutions, we use 10 images per categories (30 total). We use 50/ 50 split into members (images included in the training set) and non- members (held out images not seen during training).
- Our three gaze categories are **screen**, **paper**, and **wander**.  
a.  **screen**: the student appears to be looking at a computer screen.  
b.  **paper**: the student appears to be looking down at paper or notes on the desk.  
c.  **wander**: the student’s gaze is neither on the paper nor the screen, suggesting an off-task or wandering look. 
   
### How did you generate or collect it?
We created the dataset synthetic using a subset of an existing dataset. The dataset contains controlled image generation rather than real classroom captures.  We train a classifier (ResNet-18) to distinguish between these three gaze categories. Then we evaluate a membership inference attack (MIA), which tests whether an adversary can tell if a given image was part of the training set. This proof of concept design allows the MIA to test whether the trained model leaks enough information for an adverservary to tell which images were part of the training data.

### Provide 1-2 small examples

<img width="817" height="318" alt="{FB68E275-4036-404C-9898-772880E87594}" src="https://github.com/user-attachments/assets/006a2245-24e1-4131-ad3a-fe7f1f48fe5d" />

<img width="819" height="280" alt="{0BD81A9A-51E4-400A-846C-B17C6A69A985}" src="https://github.com/user-attachments/assets/88bb0045-ec60-49e0-a697-43cb9921f508" />


## Method/ Attack
###  What approach did you try (design choices, threat model if relevant)?
We train a ResNet-18 classifier to predict three gaze categories (**screen**, **paper**, **wander**) and then evaluate privacy leakage using a loss-based membership inference attack (*MIA*). The attack assumes an adversary with query (black-box) access to the target model and the ability to obtain the model’s loss on chosen inputs (i.e., the adversary has or can infer true labels for queried examples). We use the following core design choices:
- Use a loss-threshold MIA as a simple, interpretable baseline that directly inspects how confidently the target model fits individual examples.
- Calibrate the attack threshold on a held-out calibration set to avoid overfitting the attack to the training set.
- Report both classification metrics (accuracy, per-class F1) and MIA diagnostics (accuracy, AUC, ROC, loss distributions) to capture both utility and privacy behavior.

###  How did you implement it (tools, key steps)?
**Tools**: PyTorch (model training), torchvision (transforms / data handling), NumPy / pandas (data manipulation), scikit-learn (attack classifier / metrics), Matplotlib (plots).

**Key steps**:
1. Data preparation: assemble the synthetic dataset, create train / validation / test splits, and prepare a separate calibration set for the MIA.
2. Train target model: train ResNet-18 on the training split (augmentation and optimizer settings shown in the notebook). Save the final checkpoint and record per-example losses on train and held-out images.
3. Collect loss statistics: compute the model loss for member examples (in training set) and non-member examples (held-out set). Plot loss distributions to visualize separability.
4. Calibrate attack threshold: use the calibration set to select an optimal loss threshold (selected by maximizing calibration accuracy / AUC).
5. Evaluate MIA: apply the calibrated threshold to the evaluation set. Report attack accuracy, AUC, and ROC curve. Optionally train a simple logistic-regression attack on per-example losses and compare to the threshold baseline.
6. Document results: produce plots (loss distributions, ROC, confusion matrix) and report numerical metrics for both classification and MIA.

## Results
### Main metrics or outcomes (tables/ figures if useful).

### Classificatioan performance
The ResNet-18 classifier achieved strong performance on the synthetic dataset:
- *Final validation accuracy*: 95.83% 
- *Per- class performance (precision/ recall/ F1)):*
  a.  **paper**: 1.00/ 1.00/ 1.00 
  b.  **screen**: 1.00/ 0.88/ 0.93
  c. **wander**: 0.89/ 1.00/ 0.94
- *Overall accuracy*: 0.96
- *Marco- average F1*:0.96
Training curves as follows:

<img width="1547" height="476" alt="{09FCDB2E-1BA7-4031-B7AF-672EDBB27C43}" src="https://github.com/user-attachments/assets/1e7c19ed-205e-42ce-903a-7a135c0dead7" />

Confusion matrix:

<img width="684" height="555" alt="{26AA4019-5F81-471B-AB8C-CD3DD73B7F4D}" src="https://github.com/user-attachments/assets/b16d21d1-6969-447a-829d-9ce3aea0fe24" />

### Interpretation (1-2 sentences)
The model converges quickly and achieves high accuracy across all three classes. Most errors happen between **screen** and **wander**.  This suggest potential overlap with those gazes as shown in the dataset examples above. 

### Membership Inference Attack (MIA) Performance
<img width="1733" height="606" alt="{A972DF7F-F0A0-4E83-AFF3-EE1193E15237}" src="https://github.com/user-attachments/assets/6f65a630-9141-4735-bf42-ecebec9533f4" />

<img width="735" height="679" alt="{0D374F8E-238A-4EA8-AD66-0538803F403E}" src="https://github.com/user-attachments/assets/bdd6b9da-6108-4f1a-9792-aa160071dde3" />

### 1-2 sentences explaining what the results mean
The MIA evaluation accuracy of 0.633 and the AUC of 0.442 suggest that the attack struggles to distinguish members from non-members.  This suggest unreliable leakage evidence of the training set membership. Futher evaluation needed. 

## Implications
###  What risks, weaknesses, or lessons do these results show for your project?
- The dataset with 140 images is relatively small. High accuracy here does not necessarily mean our assessment would perform well on live classroom settings where students might show difference in postures, distractions and more naunced gaze patterns.  Also, environment factors, such as lighting and background conditions, impact results.  
- The confusion matrix shows some misclassification between **screen** and **wander**.  This suggests ovelapping features between those gazes patterns.
- The MIA results show low but non zero leakage.  The attack performed slightly better than random guessing.  This suggest exposed privacy signals.
  
###  Any key limitations or next steps
- Expand the dataset:  Include realistic variations inherest in a video versus a still image.  Other consideratins include different lighting, poses, or background.
- Stronger evaluation:  Test more advance privacy attacks (as suggested in course schedule) to better measure leakage risk.
- Model comparisons:  Experiment with other neural network architetures (eg. CNN variants, GAN) to see if the trends hold across the models.
  
## Setup requirments
- **Python version**: 3.9 or later
- **Google Colab** (the team used Colab for code review ease)
- **Libraries**:
  -  Pytorch (>=1.12) for model training
  -  torchvision for image tranforms
  -  numpy for array operations
  -  matplotlib for plotting
  -  scikit- learn for evaluation metrics   

## How to run the code
1.  Open the folder in Google Colab.
2.  Upload the dataset as zip file into the working folder director
3.  Run the cells in order from top to bottom.
4.  The notebook does the following:
    - Load the synthetic dataset that includes MIA items (three gaze categories: **screen**, **paper**, **wander**)
    - Train a ResNet-18 classifier.
    - Evaluate accuracy and confusion matrix
    - Run a membership inference attack (MIA) test.

## Expected results
The results show validation accuracy around 95%, a confusino matrix with most correct predictions, and a MIA AUC near 0.44

### Conclusion
Our experiment show that the ResNet-18 classifier distinguishes between **paper**, **screen**, and **wander** to achieve over 95% validation accuracy.  Further, the confusion matrix shows strong class separation.
However, the MIA results suggest leakage of the training data. We conjecture the matter rests with the relative closeness of the images, which itself suggests some overlapping features. For future work, we extend the dataset size, add guassian noise along with exploring stronger privacy attacks to maintain model robustness.


# Team 3: Intelligent Tutoring Systems R Us

## Dataset description.
### Overview.
This project uses a small synthetic image dataset created for a three-way classification task. The dataset simulates student attention states in a classroom or study setting. 

### What does the dataset represent (use case)?
This setup provides a simplified but relevant proxy for real-world problems in **AI in Education** (attention monitoring, adaptive feedback) while remaining lightweight enough for course assignments. We use this as a proof of concept to study **membership inference attack (MIA)**.

### How big is it (#samples, label)?
- We use 40 student images captured under three gaze categories. This reflects a sampe size of 120 images. 
- For the MIA evalutions, we use 10 images per categories (30 total). We use 50/ 50 split into members (images included in the training set) and non- members (held out images not seen during training).
- Our three gaze categories are **screen**, **paper**, and **wander**.  
a.  **screen**: the student appears to be looking at a computer screen.  
b.  **paper**: the student appears to be looking down at paper or notes on the desk.  
c.  **wander**: the student’s gaze is neither on the paper nor the screen, suggesting an off-task or wandering look. 
   
### How did you generate or collect it?
We created the dataset synthetic using a subset of an existing dataset. The dataset contains controlled image generation rather than real classroom captures.  We train a classifier (ResNet-18) to distinguish between these three gaze categories. Then we evaluate a membership inference attack (MIA), which tests whether an adversary can tell if a given image was part of the training set. This proof of concept design allows the MIA to test whether the trained model leaks enough information for an adverservary to tell which images were part of the training data.

### Provide 1-2 small examples

<img width="817" height="318" alt="{FB68E275-4036-404C-9898-772880E87594}" src="https://github.com/user-attachments/assets/006a2245-24e1-4131-ad3a-fe7f1f48fe5d" />

<img width="819" height="280" alt="{0BD81A9A-51E4-400A-846C-B17C6A69A985}" src="https://github.com/user-attachments/assets/88bb0045-ec60-49e0-a697-43cb9921f508" />


## Method/ Attack
###  What approach did you try (design choices, threat model if relevant)?
We train a ResNet-18 classifier to predict three gaze categories (**screen**, **paper**, **wander**) and then evaluate privacy leakage using a loss-based membership inference attack (*MIA*). The attack assumes an adversary with query (black-box) access to the target model and the ability to obtain the model’s loss on chosen inputs (i.e., the adversary has or can infer true labels for queried examples). We use the following core design choices:
- Use a loss-threshold MIA as a simple, interpretable baseline that directly inspects how confidently the target model fits individual examples.
- Calibrate the attack threshold on a held-out calibration set to avoid overfitting the attack to the training set.
- Report both classification metrics (accuracy, per-class F1) and MIA diagnostics (accuracy, AUC, ROC, loss distributions) to capture both utility and privacy behavior.

###  How did you implement it (tools, key steps)?
**Tools**: PyTorch (model training), torchvision (transforms / data handling), NumPy / pandas (data manipulation), scikit-learn (attack classifier / metrics), Matplotlib (plots).

**Key steps**:
1. Data preparation: assemble the synthetic dataset, create train / validation / test splits, and prepare a separate calibration set for the MIA.
2. Train target model: train ResNet-18 on the training split (augmentation and optimizer settings shown in the notebook). Save the final checkpoint and record per-example losses on train and held-out images.
3. Collect loss statistics: compute the model loss for member examples (in training set) and non-member examples (held-out set). Plot loss distributions to visualize separability.
4. Calibrate attack threshold: use the calibration set to select an optimal loss threshold (selected by maximizing calibration accuracy / AUC).
5. Evaluate MIA: apply the calibrated threshold to the evaluation set. Report attack accuracy, AUC, and ROC curve. Optionally train a simple logistic-regression attack on per-example losses and compare to the threshold baseline.
6. Document results: produce plots (loss distributions, ROC, confusion matrix) and report numerical metrics for both classification and MIA.

## Results
### Main metrics or outcomes (tables/ figures if useful).

### Classificatioan performance
The ResNet-18 classifier achieved strong performance on the synthetic dataset:
- *Final validation accuracy*: 95.83% 
- *Per- class performance (precision/ recall/ F1)):*
  a.  **paper**: 1.00/ 1.00/ 1.00 
  b.  **screen**: 1.00/ 0.88/ 0.93
  c. **wander**: 0.89/ 1.00/ 0.94
- *Overall accuracy*: 0.96
- *Marco- average F1*:0.96
Training curves as follows:

<img width="1547" height="476" alt="{09FCDB2E-1BA7-4031-B7AF-672EDBB27C43}" src="https://github.com/user-attachments/assets/1e7c19ed-205e-42ce-903a-7a135c0dead7" />

Confusion matrix:

<img width="684" height="555" alt="{26AA4019-5F81-471B-AB8C-CD3DD73B7F4D}" src="https://github.com/user-attachments/assets/b16d21d1-6969-447a-829d-9ce3aea0fe24" />

### Interpretation (1-2 sentences)
The model converges quickly and achieves high accuracy across all three classes. Most errors happen between **screen** and **wander**.  This suggest potential overlap with those gazes as shown in the dataset examples above. 

### Membership Inference Attack (MIA) Performance
<img width="1733" height="606" alt="{A972DF7F-F0A0-4E83-AFF3-EE1193E15237}" src="https://github.com/user-attachments/assets/6f65a630-9141-4735-bf42-ecebec9533f4" />

<img width="735" height="679" alt="{0D374F8E-238A-4EA8-AD66-0538803F403E}" src="https://github.com/user-attachments/assets/bdd6b9da-6108-4f1a-9792-aa160071dde3" />

### 1-2 sentences explaining what the results mean
The MIA evaluation accuracy of 0.633 and the AUC of 0.442 suggest that the attack struggles to distinguish members from non-members.  This suggest unreliable leakage evidence of the training set membership. Futher evaluation needed. 

## Implications
###  What risks, weaknesses, or lessons do these results show for your project?
- The dataset with 140 images is relatively small. High accuracy here does not necessarily mean our assessment would perform well on live classroom settings where students might show difference in postures, distractions and more naunced gaze patterns.  Also, environment factors, such as lighting and background conditions, impact results.  
- The confusion matrix shows some misclassification between **screen** and **wander**.  This suggests ovelapping features between those gazes patterns.
- The MIA results show low but non zero leakage.  The attack performed slightly better than random guessing.  This suggest exposed privacy signals.
  
###  Any key limitations or next steps
- Expand the dataset:  Include realistic variations inherent in a video versus a still image.  Other consideratins include different lighting, poses, or background.
- Stronger evaluation:  Test more advance privacy attacks (as suggested in course schedule) to better measure leakage risk.
- Model comparisons:  Experiment with other neural network architetures (eg. CNN variants, GAN) to see if the trends hold across the models.
  
## Setup requirments
- **Python version**: 3.9 or later
- **Google Colab** (the team used Colab for code review ease)
- **Libraries**:
  -  Pytorch (>=1.12) for model training
  -  torchvision for image tranforms
  -  numpy for array operations
  -  matplotlib for plotting
  -  scikit- learn for evaluation metrics   

## How to run the code
1.  Open the folder in Google Colab.
2.  Upload the dataset as zip file into the working folder director
3.  Run the cells in order from top to bottom.
4.  The notebook does the following:
    - Load the synthetic dataset that includes MIA items (three gaze categories: **screen**, **paper**, **wander**)
    - Train a ResNet-18 classifier.
    - Evaluate accuracy and confusion matrix
    - Run a membership inference attack (MIA) test.

## Expected results
The results show validation accuracy around 95%, a confusino matrix with most correct predictions, and a MIA AUC near 0.44

### Conclusion
Our experiment show that the ResNet-18 classifier distinguishes between **paper**, **screen**, and **wander** to achieve over 95% validation accuracy.  Further, the confusion matrix shows strong class separation.
However, the MIA results suggest leakage of the training data. We conjecture the matter rests with the relative closeness of the images, which itself suggests some overlapping features. For future work, we extend the dataset size, add guassian noise along with exploring stronger privacy attacks to maintain model robustness.

### How I used AI
We used ChatGPT to explain and give examples of how we can write a membership inference attack based on the example notebook.
