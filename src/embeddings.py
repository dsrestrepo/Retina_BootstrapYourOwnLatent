from .data_loader import ImageFolderDataset
from .model import FoundationalCVModel
from .RetFound import get_retfound

from torch.utils.data import DataLoader
import torch

import os
import pandas as pd
#import joblib

import numpy as np
import pandas as pd
import os

# Split
from sklearn.model_selection import train_test_split

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay

# Plots:
import matplotlib.pyplot as plt
import seaborn as sns

# Models
# Random forest
from sklearn.ensemble import RandomForestClassifier
# Logistic regression
from sklearn.linear_model import LogisticRegression
# Support vector machine
from sklearn.svm import SVC
# Decision tree
from sklearn.tree import DecisionTreeClassifier
# K-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

## Embeddings Visualization
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px

from imblearn.over_sampling import SMOTE


import warnings
warnings.filterwarnings("ignore")


# Define a function to generate embeddings in parallel
def generate_embeddings(batch, batch_number, model, device='cpu'):
    """
    Generate image embeddings for a batch of images using the specified model.

    Parameters:
    - batch (tuple): A batch of images where the first element is a list of image names, and the second element is a tensor of images.
    - batch_number (int): The batch number for tracking progress.
    - model (torch.nn.Module): The model used to generate image embeddings.

    Returns:
    tuple: A tuple containing a list of image names and their corresponding embeddings.

    Example Usage:
    ```python
    img_names, embeddings = generate_embeddings(batch, batch_number, model)
    ```

    Note:
    - This function processes a batch of images and generates embeddings for each image.
    - It is typically used in a data loading pipeline to generate embeddings for a dataset.
    """
    img_names, images = batch[0], batch[1].to(device)

    with torch.no_grad():
        features = model(images)

    if batch_number % 10 == 0:
        print(f"Processed batch number: {batch_number}")

    return img_names, features.cpu()


def get_embeddings_df(batch_size=32, path="../BRSET/images/", backbone="dinov2", directory='Embeddings', weights=None, device=None):
    """
    Generate image embeddings and save them in a DataFrame.

    Parameters:
    - batch_size (int, optional): The batch size for processing images. Default is 32.
    - path (str, optional): The path to the folder containing the images. Default is "../BRSET/images/".
    - backbone (str, optional): The name of the foundational CV model to use. Default is "dinov2".
    - directory (str, optional): The directory to save the generated embeddings DataFrame. Default is 'Embeddings'.

    Example Usage:
    ```python
    get_embeddings_df(batch_size=64, path="data/images/", backbone="vit_base", directory='output_embeddings')
    ```

    Note:
    - This function generates image embeddings for a dataset and saves them in a DataFrame.
    - The `backbone` parameter specifies the underlying model used for feature extraction.
    - The resulting DataFrame contains image names and their corresponding embeddings.

    """
    if type(backbone) == str:
        print('#'*50, f' {backbone} ', '#'*50)
    else:
        print('#'*50, f' Generating Embeddings ', '#'*50)
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the custom dataset
    shape = (224, 224)
    dataset = ImageFolderDataset(folder_path=path, shape=shape)
    
    # Create a DataLoader to generate embeddings
    batch_size = batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if type(backbone) == str:
        if backbone == 'retfound':
            model = get_retfound(weights=weights, backbone=True)
        else:
            model = FoundationalCVModel(backbone)
    else:
        model = backbone
                
    model.to(device)

    # Use DataParallel to parallelize the model across multiple GPUs
    #if torch.cuda.device_count() > 1:
        #print("Using", torch.cuda.device_count(), "GPUs!")
        #model = torch.nn.DataParallel(model, [0,1])

    img_names = []
    features = []
    for batch_number, batch in enumerate(dataloader, start=1):
        img_names_aux, features_aux = generate_embeddings(batch, batch_number, model, device)
        img_names.append(img_names_aux)
        features.append(features_aux)

    """
    # Parallelize the embedding generation process using joblib
    results = joblib.Parallel(n_jobs=-1, prefer="threads")(
        joblib.delayed(generate_embeddings)(batch, batch_number)
        for batch_number, batch in enumerate(dataloader, start=1)
    )
    """

    # Flatten the results to get a list of image names and their corresponding embeddings
    all_img_names = [item for sublist in img_names for item in sublist]
    all_embeddings = [item.tolist() for sublist in features for item in sublist]

    # Create a DataFrame with image names and embeddings
    df = pd.DataFrame({
        'ImageName': all_img_names,
        'Embeddings': all_embeddings
    })
    

    df_aux = pd.DataFrame(df['Embeddings'].tolist())
    df = pd.concat([df['ImageName'], df_aux], axis=1)

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if type(backbone) == str:
        df.to_csv(f'{directory}/Embeddings_{backbone}.csv', index=False)
    else:
        df.to_csv(f'{directory}/Embeddings.csv', index=False)
    


def load_data(labels_path='data/labels.csv', backbone='dinov2_large', label='diabetic_retinopathy', directory='Embeddings', dataset_name='BRSET', normal=False, DR_ICDR_3=True, extra_labels=None, quality=False):
    """
    Load and prepare data for a machine learning task using image embeddings and corresponding labels.

    This function loads image embeddings from a CSV file and merges them with label data from another CSV file.
    The function allows for data preprocessing and label encoding for specific classification tasks.

    Parameters:
    - labels_path (str, optional): The path to the CSV file containing label data. Default is 'data/labels.csv'.
    - backbone (str, optional): The name of the backbone used for image embeddings. Default is 'dinov2_large'.
    - label (str, optional): The label to use for classification. Default is 'diabetic_retinopathy'.
    - directory (str, optional): The directory where the embeddings CSV file is located. Default is 'Embeddings'.
    - normal (bool, optional): If True, filter data for normal samples. Default is False.

    Returns:
    - X (pandas.DataFrame): Feature data (image embeddings) for the machine learning task.
    - y (pandas.Series): Target data (labels) for the machine learning task.

    Example Usage:
    ```python
    X, y = load_data(labels_path='data/labels.csv', backbone='dinov2_large', label='diabetic_retinopathy', directory='Embeddings', normal=False)
    ```

    Note:
    - This function is designed to load image embeddings and corresponding labels, allowing for data preprocessing and label encoding.
    - It is suitable for machine learning tasks where you have image features and labels.

    """

    # Embeddings
    if type(backbone) == str:
        embeddings_path = f'{directory}/Embeddings_{backbone}.csv'
    else:
        embeddings_path = f'{directory}/Embeddings.csv'
        
    df = pd.read_csv(embeddings_path)
    df.rename(columns={'ImageName':'image_id'}, inplace=True)
    df['image_id'] = df['image_id'].apply(lambda x: x.replace('.jpg',''))
    df['image_id'] = df['image_id'].apply(lambda x: x.replace('.JPG',''))
    df['image_id'] = df['image_id'].apply(lambda x: x.replace('.png',''))
    df['image_id'] = df['image_id'].apply(lambda x: x.replace('.jpeg',''))
    
    if (label == 'DR_ICDR') and (dataset_name.lower() == 'eyepacs'):
        label = 'level'
    elif (label == 'DR_ICDR') and ('messidor' in dataset_name.lower()):
        label = 'diagnosis'
    
    if dataset_name == 'BRSET':
        # Labels
        brset_df = pd.read_csv(labels_path)
        if normal:
            brset_df = brset_df[brset_df['DR_ICDR'] == 0]

        brset_df['patient_age'].fillna(brset_df['patient_age'].mean(), inplace=True)
        # One-hot encode categorical variables:
        brset_df = pd.get_dummies(brset_df, columns=['camera', 'optic_disc', 'diabetes'])
        
    elif ('messidor' in dataset_name.lower()):
        # Labels
        brset_df = pd.read_csv(labels_path)
        brset_df['image_id'] = brset_df['image_id'].apply(lambda x: x.replace('.jpg',''))
        brset_df['image_id'] = brset_df['image_id'].apply(lambda x: x.replace('.JPG',''))
        brset_df['image_id'] = brset_df['image_id'].apply(lambda x: x.replace('.png',''))
        brset_df['image_id'] = brset_df['image_id'].apply(lambda x: x.replace('.jpeg',''))
        
    elif dataset_name.lower() == 'eyepacs':
        # Labels
        brset_df = pd.read_csv(labels_path)
        brset_df.rename(columns={'image':'image_id'}, inplace=True)
        brset_df['image_id'] = brset_df['image_id'].apply(lambda x: x.replace('.jpg',''))
        brset_df['image_id'] = brset_df['image_id'].apply(lambda x: x.replace('.JPG',''))
        brset_df['image_id'] = brset_df['image_id'].apply(lambda x: x.replace('.png',''))
        brset_df['image_id'] = brset_df['image_id'].apply(lambda x: x.replace('.jpeg',''))
        

    # Merge
    df = brset_df.merge(df, on='image_id')
    
    # remove image field of quiality: quality == 'Adequate if 'focus', 'iluminaton', 'artifacts' == 2; quality = 'Inadequate if 'focus', 'iluminaton', 'artifacts' == 1
    df['quality'] = df.apply(lambda x: 'Adequate' if (x['focus'] == 1) or (x['iluminaton'] == 1) or (x['artifacts'] == 1) else 'Inadequate', axis=1)
    
    if quality == 'focus':
        df['quality'] = df.apply(lambda x: 'Adequate' if (x['focus'] == 1) else 'Inadequate', axis=1)
    elif quality == 'iluminaton':
        df['quality'] = df.apply(lambda x: 'Adequate' if (x['iluminaton'] == 1) else 'Inadequate', axis=1)
    elif quality == 'artifacts':
        df['quality'] = df.apply(lambda x: 'Adequate' if (x['artifacts'] == 1) else 'Inadequate', axis=1)
        
    
    
    if quality:
        df_quality = df[df['quality'] == 'Adequate']
        df_bad_quality = df[df['quality'] == 'Inadequate']
        y_quality = df_quality[label]
        y_bad_quality = df_bad_quality[label]
        X_quality = df_quality.iloc[:, brset_df.shape[1]:]
        X_bad_quality = df_bad_quality.iloc[:, brset_df.shape[1]:]
        
        # Check if extra_labels is a list and not None
        if extra_labels is not None and isinstance(extra_labels, list):
            # Select the specified extra_labels columns from the DataFrame and append them to the features set (X)
            extra_labels_df = df_quality[extra_labels]
            X_quality = pd.concat([X_quality, extra_labels_df], axis=1)
            
            # Select the specified extra_labels columns from the DataFrame and append them to the features set (X)
            extra_labels_df = df_bad_quality[extra_labels]
            X_bad_quality = pd.concat([X_bad_quality, extra_labels_df], axis=1)
        
        if label in ['DR_ICDR', 'diagnosis', 'level']:
            if DR_ICDR_3:
                # 0: Normal = 0
                # 1, 2, 3 Non-proliferative = 1
                # 4 Proliferative = 2
                y_quality = y_quality.apply(lambda x: 1 if x in (1, 2, 3) else (2 if x == 4 else 0))
                y_bad_quality = y_bad_quality.apply(lambda x: 1 if x in (1, 2, 3) else (2 if x == 4 else 0))
            else:
                pass
        if dataset_name == 'BRSET':
            if label == 'diabetes':
                y_quality = y_quality.apply(lambda x: 1 if x == 'yes' else 0)
                y_bad_quality = y_bad_quality.apply(lambda x: 1 if x == 'yes' else 0)
            if label == 'patient_sex':
                y_quality = y_quality.apply(lambda x: 0 if x == 2 else 1) 
                y_bad_quality = y_bad_quality.apply(lambda x: 0 if x == 2 else 1)
        return X_quality, y_quality, X_bad_quality, y_bad_quality
    else:    
        y = df[label]
        X = df.iloc[:, brset_df.shape[1]:]
        
        # Check if extra_labels is a list and not None
        if extra_labels is not None and isinstance(extra_labels, list):
            # Select the specified extra_labels columns from the DataFrame and append them to the features set (X)
            extra_labels_df = df[extra_labels]
            X = pd.concat([X, extra_labels_df], axis=1)
        
        if label in ['DR_ICDR', 'diagnosis', 'level']:
            if DR_ICDR_3:
                # 0: Normal = 0
                # 1, 2, 3 Non-proliferative = 1
                # 4 Proliferative = 2
                y = y.apply(lambda x: 1 if x in (1, 2, 3) else (2 if x == 4 else 0))
            else:
                pass
        
        if dataset_name == 'BRSET':        
            if label == 'diabetes':
                y = y.apply(lambda x: 1 if x == 'yes' else 0)

            if label == 'patient_sex':
                y = y.apply(lambda x: 0 if x == 2 else 1) 
        
        return X, y


def split_dataset(X, y, test_size=0.3, random_state=1, plot=True, oversample=False):
    """
    Split a dataset into training and testing sets and optionally visualize class distribution.

    This function takes feature data (X) and corresponding labels (y) and splits them into training and testing sets using the
    train_test_split function from scikit-learn. It also provides an option to visualize the class distribution in both sets.

    Parameters:
    - X (numpy.ndarray or pandas.DataFrame): The feature data to be split.
    - y (numpy.ndarray or pandas.Series): The corresponding labels for the feature data.
    - test_size (float, optional): The proportion of the dataset to include in the test split (default is 0.3).
    - random_state (int, optional): The random seed for reproducible splitting (default is 1).
    - plot (bool, optional): Whether to plot class distribution in training and test sets (default is True).

    Returns:
    - X_train (numpy.ndarray or pandas.DataFrame): The feature data for the training set.
    - X_test (numpy.ndarray or pandas.DataFrame): The feature data for the test set.
    - y_train (numpy.ndarray or pandas.Series): The labels for the training set.
    - y_test (numpy.ndarray or pandas.Series): The labels for the test set.

    Example Usage:
    ```python
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=42, plot=True)
    ```

    Note:
    - The function uses scikit-learn's train_test_split to split the data into training and test sets.
    - If the 'plot' parameter is set to True, class distribution bar plots for both sets are displayed.

    Dependencies:
    - numpy or pandas for feature data and labels
    - scikit-learn (train_test_split) for data splitting
    - matplotlib for data visualization (if 'plot' is True)
    """
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    print(f"Training set size is: {len(X_train)} rows and {X_train.shape[1]} columns")
    print(f"Test set size is: {len(X_test)} rows and {X_test.shape[1]} columns")
    
    if oversample:
        # Apply SMOTE to oversample the minority class in the training set
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("Applied SMOTE to oversample the training set.")
    
    if plot:

        # Calculate class distribution in the training and test sets
        train_class_counts = np.bincount(y_train)
        test_class_counts = np.bincount(y_test)

        # Get the unique class labels
        unique_labels = np.unique(np.concatenate((y_train, y_test)))
        
        if len(unique_labels) != len(train_class_counts):
            print('There are missing classes in the training set')
            print('Available classes:', unique_labels)
            return None, None, None, None
        
        # Create bar plots to visualize class distribution
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(unique_labels, train_class_counts)
        plt.title("Class Distribution in Training Set")
        plt.xlabel("Class Label")
        plt.ylabel("Count")
        plt.xticks(unique_labels)

        plt.subplot(1, 2, 2)
        plt.bar(unique_labels, test_class_counts)
        plt.title("Class Distribution in Test Set")
        plt.xlabel("Class Label")
        plt.ylabel("Count")
        plt.xticks(unique_labels)

        plt.tight_layout()
        plt.show()
    
    return X_train, X_test, y_train, y_test


def visualize_embeddings(X_train, X_test, y_train, y_test, plot_type='2D', method='UMAP'): 
    """
    Visualize high-dimensional data embeddings in 2D or 3D space using various dimensionality reduction techniques.

    Parameters:
    - X_train (array-like): Training data embeddings or feature vectors.
    - X_test (array-like): Test data embeddings or feature vectors.
    - y_train (array-like): Labels or target values for the training data.
    - y_test (array-like): Labels or target values for the test data.
    - plot_type (str, optional): The type of visualization, either '2D' or '3D'. Default is '2D'.
    - method (str, optional): The dimensionality reduction technique to use ('PCA', 't-SNE', or 'UMAP'). Default is 'UMAP'.

    Returns:
    None

    Example Usage:
    ```python
    visualize_embeddings(X_train, X_test, y_train, y_test, plot_type='2D', method='UMAP')
    ```

    Note:
    - This function allows you to visualize high-dimensional data embeddings in 2D or 3D space using different dimensionality reduction methods.
    - It is particularly useful for understanding the structure and patterns within the data.
    - You can choose the type of visualization ('2D' or '3D') and the dimensionality reduction technique ('PCA', 't-SNE', or 'UMAP').

    Dependencies:
    - NumPy
    - Scikit-learn (for PCA and t-SNE)
    - UMAP-learn (for UMAP)
    - Plotly (for interactive visualization)

    For more information on the dimensionality reduction techniques and their parameters, refer to the respective documentation.
    """


    perplexity = 10

    if plot_type == '3D':
        if method == 'PCA':
            # Apply PCA to reduce the dimensionality of the embeddings
            red = PCA(n_components=3, random_state=42).fit(X_train)
            reduced_embeddings = red.transform(X_test)
        elif method == 't-SNE':
            # Apply t-SNE to reduce the dimensionality of the embeddings
            red = TSNE(n_components=3, perplexity=perplexity, random_state=42).fit(X_train)
            reduced_embeddings = red.transform(X_test)
        elif method == 'UMAP':
            from umap import UMAP
            # Apply UMAP to reduce the dimensionality of the embeddings
            red = UMAP(n_components=3, random_state=42).fit(X_train)
            reduced_embeddings = red.transform(X_test)
            
        # Combine df1 with the UMAP results
        df_reduced = pd.DataFrame(reduced_embeddings, columns=['col1', 'col2', 'col3'])
        # Combine df_reduced with y
        df_reduced['Class'] = y_test.reset_index().iloc[:, 1]
        
        # Create 2D and 3D scatter plots with Plotly
        fig_3d = px.scatter_3d(df_reduced, x='col1', y='col2', z='col3', color='Class', title='3D UMAP')

    else:
        if method == 'PCA':
            # Apply PCA to reduce the dimensionality of the embeddings
            red = PCA(n_components=2, random_state=42).fit(X_train)
            reduced_embeddings = red.transform(X_test)
        elif method == 't-SNE':
            # Apply t-SNE to reduce the dimensionality of the embeddings
            red = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit(X_train)
            reduced_embeddings = red.transform(X_test)
        elif method == 'UMAP':
            from umap import UMAP
            # Apply UMAP to reduce the dimensionality of the embeddings
            red = UMAP(n_components=2, random_state=42).fit(X_train)
            reduced_embeddings = red.transform(X_test)
        
        # Combine df1 with the results
        df_reduced = pd.DataFrame(reduced_embeddings, columns=['col1', 'col2'])
        
        
        # Combine df_reduced with y
        df_reduced['Class'] = y_test.reset_index().iloc[:, 1]

        # Create 2D and 3D scatter plots with Plotly
        fig = px.scatter(df_reduced, x='col1', y='col2', color='Class', title='2D UMAP')
    
    # Set plot layout
    fig.update_layout(
        title=f"Embeddings - {method} {plot_type} Visualization",
        scene=dict(
        )
    )
    
    # Show the interactive plot
    fig.show()

def test_model(X_test, y_test, model):
    """
    Evaluates the model on the training and test data respectively
    1. Predictions on test data
    2. Classification report
    3. Confusion matrix
    4. ROC curve

    Inputs:
    X_test: numpy array with test features
    y_test: numpy array with test labels
    model: trained model
    """
    
    # Predictions on test data
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    
    if not isinstance(y_test, pd.Series):
        print('Y_test is not a pandas Series')
        y_test_series = pd.Series(y_test, index=X_test.index)
    else:
        y_test_series = y_test

    # Convert y_pred to a pandas Series using the same index as y_test_series
    y_pred_series = pd.Series(y_pred, index=y_test_series.index)

    # Identify indices of wrong predictions
    wrong_predictions = y_test_series != y_pred_series
    wrong_indices = wrong_predictions[wrong_predictions].index
    
    

    # Confusion matrix
    # Create a confusion matrix of the test predictions
    cm = confusion_matrix(y_test, y_pred)
    # create heatmap
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    # set plot labels
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    # display plot
    plt.show()

    #create ROC curve
    from sklearn.preprocessing import LabelBinarizer
    fig, ax = plt.subplots(figsize=(6, 6))

    label_binarizer = LabelBinarizer().fit(y_test)
    y_onehot_test = label_binarizer.transform(y_test)
    y_onehot_pred = label_binarizer.transform(y_pred)
    
    if (y_onehot_pred.shape[1] < 2):
        fpr, tpr, _ = roc_curve(y_test,  y_pred)

        #create ROC curve
        #plt.plot(fpr,tpr)
        RocCurveDisplay.from_predictions(
                y_test,
                y_prob[:, 1],
                name=f"ROC curve",
                color='aqua',
                ax=ax,
            )
        plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
        plt.title('ROC Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        
    else:
        from itertools import cycle
        colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "yellow", "purple", "pink", "brown", "black"])

        for class_id, color in zip(range(len(label_binarizer.classes_)), colors):
            RocCurveDisplay.from_predictions(
                y_onehot_test[:, class_id],
                y_prob[:, class_id],
                name=f"ROC curve for {label_binarizer.classes_[class_id]}",
                color=color,
                ax=ax,
                
            )

        plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
        plt.legend()
        plt.show()
        
    # Classification report
    # Create a classification report of the test predictions
    cr = classification_report(y_test, y_pred)
    # print classification report
    print(cr)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class precision
    recall = recall_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class recall
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class F1-score

    return accuracy, precision, recall, f1, wrong_indices


def train_and_evaluate_model(X_train, X_test, y_train, y_test, models=None):
    """
    Trains and evaluates multiple classification models
    The list of models to train can be specified, otherwise a default list is used
    Default list: 
    - Random Forest
    -Decision Tree
    -Logistic Regression
    -KNN
    -SVM
    
    Inputs:
    X_train: numpy array with training features
    X_test: numpy array with test features
    y_train: numpy array with training labels
    y_test: numpy array with test labels
    models: list of tuples with model name and model object
    train: boolean to indicate whether to train the models or not

    Output:
    models: list of trained models
    """
    
    visualize_embeddings(X_train, X_test, y_train, y_test, plot_type='2D', method='UMAP')
    
    # Train and evaluate multiple classification models
    if not(models):
        models = [
            ('Random Forest', RandomForestClassifier()),
            ('Decision Tree', DecisionTreeClassifier()),
            ('Logistic Regression', LogisticRegression()),
            ('KNN', KNeighborsClassifier()),
            ('SVM', SVC())
        ]

    for name, model in models:
        
        print('#'*20, f' {name} ', '#'*20)
        
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the testing set
        accuracy, precision, recall, f1, wrong_indices = test_model(X_test, y_test, model)
        
        #print(f"Accuracy: {accuracy:.2f}")
        #print(f"Precision: {precision:.2f}")
        #print(f"Recall: {recall:.2f}")
        #print(f"F1 Score: {f1:.2f}")
        
        #print('#'*80)
        
    return models, wrong_indices
"""
def test_model_nn(X_test, y_test, model):
    # Predictions on test data
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    wrong_indices = np.where(y_test != y_pred)[0]
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve for binary classification
    if y_pred_probs.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_probs[:, 1])
        plt.plot(fpr, tpr, color='aqua')
        plt.plot([0, 1], [0, 1], linestyle='--', color='black')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1, wrong_indices
    
import tensorflow as tf

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
               -tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0))
    return focal_loss_fixed


def train_and_evaluate_model_nn(X_train, X_test, y_train, y_test, input_dim, num_classes, layers=[128, 64], dropout=0.2, epochs=10, batch_size=32, class_weights=None, loss='focal'):
    # Convert labels to categorical one-hot encoding
    y_train_onehot = to_categorical(y_train, num_classes=num_classes)
    y_test_onehot = to_categorical(y_test, num_classes=num_classes)
    
    # Each blok is dense relu, batch normalization, and dropout
    model = Sequential()
    for layer in layers:
        model.add(Dense(layer, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
    
    model.add(Dense(num_classes, activation='softmax'))
    

    # Compile the model
    if loss == 'focal':
        model.compile(optimizer='adam',
                        loss=focal_loss(),
                        metrics=['accuracy'])
    else:
        model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    # Train the model
    if class_weights:
        model.fit(X_train, y_train_onehot, epochs=10, batch_size=32, validation_split=0.2, class_weight=class_weights)
    else:
        model.fit(X_train, y_train_onehot, epochs=10, batch_size=32, validation_split=0.2)
    
    # Evaluate the model
    results = test_model_nn(X_test, y_test, model)

    return model, results


"""





import torch
import torch.nn as nn
import torch.nn.functional as F
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_classes, dropout=0.2):
        super(Classifier, self).__init__()
        layers = []

        for hidden in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.BatchNorm1d(hidden))
            input_dim = hidden

        # Final layer
        layers.append(nn.Linear(hidden_layers[-1], num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def test_model_nn(X_test, y_test, model, device):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test).float().to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

    y_pred = predicted.cpu().numpy()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(y_test, y_pred))

    return accuracy, precision, recall, f1

def train_model(model, criterion, optimizer, train_loader, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    print('Finished Training')

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()



def train_and_evaluate_model_nn(X_train, X_test, y_train, y_test, input_dim, num_classes, layers=[128, 64], dropout=0.2, epochs=20, batch_size=128, class_weights=None, loss='cross_entropy'):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = Classifier(input_dim=input_dim, hidden_layers=layers, num_classes=num_classes, dropout=dropout).to(device)

    #if loss == 'focal_loss':
    #    criterion = FocalLoss(gamma=2, alpha=class_weights).to(device)
    #else:
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert datasets to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values).float()
    y_train_tensor = torch.tensor(y_train.values).long()
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test.values).float(), torch.tensor(y_test.values).long())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_model(model, criterion, optimizer, train_loader, device, epochs=epochs)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    # Evaluation
    accuracy = accuracy_score(y_test, all_preds)
    precision = precision_score(y_test, all_preds, average='weighted')
    recall = recall_score(y_test, all_preds, average='weighted')
    f1 = f1_score(y_test, all_preds, average='weighted')

    # Confusion Matrix
    cm = confusion_matrix(y_test, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(y_test, all_preds))

    return model, (accuracy, precision, recall, f1)