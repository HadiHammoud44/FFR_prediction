import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights
    
# class MIPredictor(nn.Module):
#     def __init__(self, concept_dim, hidden_dim=32, dropout_rate=0.4):
#         super().__init__()

#         self.layers = nn.Sequential(
#             nn.Linear(concept_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
            
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
            
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, concepts):
#         MI = self.layers(concepts)
#         return MI

class IndvConceptPredictor_Regression(nn.Module):
    ''' Concept Predictor Model '''
                                                                    
    def __init__(self, use_clinical_data=False, clinical_data_dim=0, embedding_size=32,dropout_p=0.3):
        super().__init__()

        self.use_clinical_data = use_clinical_data
        self.dropout = nn.Dropout(dropout_p)
        self.relu = nn.ReLU()
        
        # Get resnet18
        self.resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=ResNet18_Weights.IMAGENET1K_V1)

        # Swap the first layer to input 2 channels instead of 3
        self.resnet18.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Swap out the fully connected layer to a custom one (but first negate the predefined)
        self.resnet18.fc = nn.Identity()
        
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_p),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64 + clinical_data_dim, 32),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(32, embedding_size),
        )
        
        self.out = nn.Linear(embedding_size, 1)

    def forward(self, images, clinical_data=None):
        embedding = self.resnet18(images)
        embedding = self.fc1(embedding)

        # Use the classifier along with the clinical data (if use_clinical_data=True)
        if self.use_clinical_data:
            embedding = torch.hstack([embedding, clinical_data])

        # Apply the fc2 block to get the embedding
        embedding = self.fc2(embedding)

        # Apply ReLU activation and dropout
        concept_pred = self.out(self.dropout(self.relu(embedding)))

        return concept_pred, embedding

