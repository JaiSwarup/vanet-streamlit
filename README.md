
# VANET Packet Delivery Prediction  

## Objective  
The goal of this project is to predict whether a data packet in a **Vehicular Ad Hoc Network (VANET)** will successfully reach its destination. This is a **binary classification task** based on a dataset of simulated VANET metrics collected in urban scenarios.  

The project leverages a **deep learning-based neural network** model to analyze the following features:  
- Available Bandwidth (ABE)  
- Trajectory of the vehicle (TJR)  
- Number of Neighbors (NV)  
- Distance to Destination (DST)  
- MAC Layer Losses (LMAC)  

## Dataset Description  
The dataset includes **5823 samples** with the following columns:  
1. **ABE**: Available bandwidth in the link.  
2. **TJR**: Trajectory of the vehicle regarding the destination.  
3. **NV**: Number of neighbors.  
4. **DST**: Distance to the destination.  
5. **LMAC**: MAC layer losses.  
6. **OUT**: Target variable (1 for successful delivery, 0 otherwise).  

The dataset has been preprocessed and normalized for training and evaluation purposes.  

## Methodology  
1. **Data Preprocessing**  
   - Data is normalized to ensure uniformity across features.  
   - The dataset is split into **training (80%)** and **testing (20%)** sets.  

2. **Exploratory Data Analysis (EDA)**  
   - Correlation heatmap to understand feature relationships.  
   - Distribution plots for target variable (`OUT`).  
   - Dataset preview and summary statistics.  

3. **Model Architecture**  
   - A **Sequential Neural Network** with the following layers:  
     - Fully connected layers with ReLU activation:  
       - 256 -> 128 -> 64 -> 32 units.  
     - Output layer with sigmoid activation for binary classification.  
   - Optimized using the **Adam** optimizer with a **binary cross-entropy loss function**.  

4. **Evaluation Metrics**  
   - The model's performance is evaluated using the following metrics:  
     - **Accuracy**  = 0.8215
     - **Precision**  = 0.7996
     - **Recall**  = 0.7870
     - **F1 Score** = 0.7932
   - A **confusion matrix** is plotted for visual analysis of classification results.  

5. **Deployment**  
   - A **Streamlit application** is built with two tabs:  
     - **Explore**: Displays dataset preview, EDA graphs, and model metrics.  
     - **Predict**: Allows users to input feature values and predict the packet delivery outcome.  

## How to Run  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/JaiSwarup/vanet-streamlit.git  
   cd vanet-prediction  
   ```  

2. Install the required dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. Train the model or use the pre-trained saved model:  
   - Training: Run the training script to train the neural network.  
     ```bash  
     python train_model.py  
     ```  
   - Using Pre-trained Model: Ensure `model.h5` is in the working directory.  

4. Run the Streamlit app:  
   ```bash  
   streamlit run app.py  
   ```  


Link to the original dataset - https://upcommons.upc.edu/handle/2117/353774
