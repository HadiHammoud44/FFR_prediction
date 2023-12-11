import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import wandb 
from seq_model_reg import IndvConceptPredictor_Regression
from torch.utils.data import DataLoader, TensorDataset


def build_combined_loader(batch_size, device, data_path='data/patches'):
    # Load training data
    images = torch.load(f'{data_path}/patch_fold1/train/imgs.pt').to(device)
    labels = torch.load(f'{data_path}/patch_fold1/train/labels.pt').to(device)
    clinical_data = torch.load(f'{data_path}/patch_fold1/train/features.pt').to(device)

    # Create DataLoader for combined data
    combined_data = TensorDataset(images, labels, clinical_data)
    data_loader = DataLoader(combined_data, batch_size=batch_size, shuffle=True)

    return data_loader

def build_test_loader(device, data_path='data/patches'):
   
    #Load data for the current fold 
    images = torch.load(f'{data_path}/patch_fold1/test/imgs.pt').to(device)
    labels = torch.load(f'{data_path}/patch_fold1/test/labels.pt').to(device)
    clinical_data = torch.load(f'{data_path}/patch_fold1/test/features.pt').to(device)
    
    # Create DataLoader
    combined_data = TensorDataset(images, labels, clinical_data)
    data_loader = DataLoader(combined_data, batch_size=256, shuffle=False)

    return data_loader


def train_epoch_IndvConcept(model, data_loader, training_for, optimizer, device):   
    model.train()
    cum_concept_loss = 0
        
    criterion = nn.L1Loss(reduction='sum')

    for batch_data in data_loader:
        optimizer.zero_grad()  # Zero the gradients

        batch_images, batch_labels, batch_clinical_data = batch_data
        batch_images, batch_labels, batch_clinical_data = batch_images.to(device), batch_labels.to(device).to(torch.float), batch_clinical_data.to(device)

                                        
        batch_concept = batch_labels[:,:1] if (training_for=='DS') else batch_labels[:, 1:2] 

        # Forward pass
        concept_pred, _ = model(batch_images, batch_clinical_data)

        # Calculate the concept loss
        concept_loss = criterion(concept_pred, batch_concept) 
            
        cum_concept_loss += concept_loss.item()

        # Backward pass and optimization
        concept_loss.backward()
        optimizer.step()

    return cum_concept_loss / len(data_loader.dataset)

def test_IndvConcept(model, data_loader, training_for, device):    
    model.eval()

    # Initialize metrics
    metric = nn.L1Loss(reduction='sum')
    
    cum_concept_loss = 0

    with torch.no_grad():
        for batch_data in data_loader:
            batch_images, batch_labels, batch_clinical_data = batch_data
            batch_images, batch_labels, batch_clinical_data = batch_images.to(device), batch_labels.to(device).to(torch.float), batch_clinical_data.to(device)

            batch_concept = batch_labels[:,:1] if (training_for=='DS') else batch_labels[:, 1:2] 

            # Forward pass
            concept_pred, _ = model(batch_images, batch_clinical_data)
            
            # calculate the validation loss
            cum_concept_loss += metric(concept_pred, batch_concept).item()

    return cum_concept_loss/len(data_loader.dataset)

# training_for = 'FFR' 
# dropout_p = 0.2
# learning_rate = 0.01
# weight_decay = 0.0001 ### this is the regularization weight
# I did not make use of the schedular  
def traintest_IndvConcept(training_for, dropout_p, learning_rate, weight_decay, batch_size=64, scheduler_step_size=100, scheduler_gamma=1, num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # reset the model for the new fold
    model = IndvConceptPredictor_Regression(use_clinical_data=False, clinical_data_dim=0, embedding_size=32, dropout_p=dropout_p).to(device) 
    
    train_loader = build_combined_loader(batch_size, device)
    test_loader = build_test_loader(device)
                                 
    # Define optimizer based on current learning rate
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Define learning rate scheduler
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    train_losses = []
    
    ###temp:
    test_losses = []

    for epoch in range(num_epochs):
        
        train_loss = train_epoch_IndvConcept(model, train_loader, training_for, optimizer, device)
        train_losses.append(train_loss)
        
        ### temp
        test_losses.append(test_IndvConcept(model, test_loader, training_for, device))
        
        # Step the scheduler
        scheduler.step()
        
    torch.save(model.state_dict(), f'{training_for}_PredictorRegression_{num_epochs}epochs.pt')
    # test_loss = test_IndvConcept(model, test_loader, training_for, device)

    return train_losses, test_losses
