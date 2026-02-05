from torch import nn


class IMUDVLCNN(nn.Module):
    def __init__(self):
        super(IMUDVLCNN, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        # Use adaptive pooling to always get a fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(12)

        self.fc1 = nn.Linear(256 * 12, 512)
        self.fc2 = nn.Linear(512, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, time, features] -> [batch, features, time]

        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))

        # Use adaptive pooling instead of regular pooling
        x = self.adaptive_pool(x)  # Always outputs [batch, 256, 12]

        x = x.view(-1, 256 * 12)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # # Define the CNN model
    # class IMUDVLCNN(nn.Module):
    #     def __init__(self, dropout_rate=0.2):  # Reduced dropout rate
    #         super(IMUDVLCNN, self).__init__()
    #         # Increase network capacity slightly
    #         self.bn1 = nn.BatchNorm1d(128)
    #         self.bn2 = nn.BatchNorm1d(256)
    #         self.bn3 = nn.BatchNorm1d(512)
    #
    #         self.conv1 = nn.Conv1d(6, 128, kernel_size=5, padding=1)
    #         self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=1)
    #         self.conv3 = nn.Conv1d(256, 512, kernel_size=5, padding=1)
    #
    #         self.pool = nn.AdaptiveAvgPool1d(1)
    #         self.dropout = nn.Dropout(dropout_rate)
    #
    #         self.fc1 = nn.Linear(512, 1024)
    #         self.fc2 = nn.Linear(1024, 3)
    #         self.relu = nn.ReLU()
    #
    #     def forward(self, x):
    #         x = x.permute(0, 2, 1)
    #
    #         # Residual connection for first block
    #         identity = self.conv1(x)
    #         x = self.conv1(x)
    #         x = self.bn1(x)
    #         x = self.relu(x)
    #         x = x + identity  # Residual connection
    #         x = self.dropout(x)
    #
    #         # Second block
    #         x = self.conv2(x)
    #         x = self.bn2(x)
    #         x = self.relu(x)
    #         x = self.dropout(x)
    #
    #         # Third block
    #         x = self.conv3(x)
    #         x = self.bn3(x)
    #         x = self.relu(x)
    #         x = self.dropout(x)
    #
    #         x = self.pool(x)
    #         x = x.view(x.size(0), -1)
    #
    #         x = self.fc1(x)
    #         x = self.relu(x)
    #         x = self.dropout(x)
    #         x = self.fc2(x)
    #
    #         return x