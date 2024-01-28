import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from model import IndvConceptPredictor
from torch.utils.data import DataLoader, TensorDataset
import argparse
import json


def build_train_loader(batch_size, device, fold, data_path='data/patches'):
    # Load training data
    images = torch.load(f'{data_path}/patch_fold{fold}/train/imgs.pt').to(device)
    labels = torch.load(f'{data_path}/patch_fold{fold}/train/labels.pt').to(device)
    clinical_data = torch.load(f'{data_path}/patch_fold{fold}/train/features.pt').to(device)

    # Create DataLoader for training data
    train_data = TensorDataset(images, labels, clinical_data)
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return data_loader

def build_test_loader(device, fold, data_path='data/patches'):
   
    #Load data for the current fold 
    images = torch.load(f'{data_path}/patch_fold{fold}/test/imgs.pt').to(device)
    labels = torch.load(f'{data_path}/patch_fold{fold}/test/labels.pt').to(device)
    clinical_data = torch.load(f'{data_path}/patch_fold{fold}/test/features.pt').to(device)
    
    # Create DataLoader
    test_data = TensorDataset(images, labels, clinical_data)
    data_loader = DataLoader(test_data, batch_size=256, shuffle=False)

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
 
def traintest_IndvConcept(training_for, dropout_p, learning_rate, weight_decay, batch_size=64, scheduler_step_size=100, scheduler_gamma=1, num_epochs=25, fold=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # reset the model for the new fold
    model = IndvConceptPredictor(use_clinical_data=False, clinical_data_dim=0, embedding_size=32, dropout_p=dropout_p).to(device) 
    
    train_loader = build_train_loader(batch_size, device, fold)
    test_loader = build_test_loader(device, fold)
                                 
    # Define optimizer based on current learning rate
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Define learning rate scheduler
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    train_losses = []
    
    test_losses = []

    for epoch in range(num_epochs):
        
        train_loss = train_epoch_IndvConcept(model, train_loader, training_for, optimizer, device)
        train_losses.append(train_loss)
        
        test_losses.append(test_IndvConcept(model, test_loader, training_for, device))
        
        # Step the scheduler
        scheduler.step()
        
    # torch.save(model.state_dict(), f'{training_for}_PredictorRegression_{num_epochs}epochs.pt')
    # test_loss = test_IndvConcept(model, test_loader, training_for, device)

    return train_losses, test_losses


def main():
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--training_for', type=str, default='DS', help='Specify the training concept')
    parser.add_argument('--dropout_p', type=float, default=0.2, help='Specify dropout probability')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Specify learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Specify weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Specify batch size')
    parser.add_argument('--scheduler_step_size', type=int, default=100, help='Specify scheduler step size')
    parser.add_argument('--scheduler_gamma', type=float, default=1, help='Specify scheduler gamma')
    parser.add_argument('--num_epochs', type=int, default=25, help='Specify the number of epochs')
    parser.add_argument('--fold', type=int, default=1, help='Specify the fold')

    args = parser.parse_args()

    # Call your function with the specified arguments
    train_losses, test_losses = traintest_IndvConcept(
        args.training_for, args.dropout_p, args.learning_rate,
        args.weight_decay, args.batch_size, args.scheduler_step_size,
        args.scheduler_gamma, args.num_epochs, args.fold
    )
    
    data = {
    'train_losses': train_losses,
    'test_losses': test_losses
    }

    # Save the dictionary to a JSON file
    output_filename = f"loss_data_fold{args.fold}.json"
    with open(output_filename, 'w') as file:
        json.dump(data, file)

if __name__ == "__main__":
    main()