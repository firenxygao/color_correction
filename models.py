import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP3(nn.Module):
    def __init__(self):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(3, 79)
        self.fc2 = nn.Linear(79, 36)
        self.fc3 = nn.Linear(36, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP5(nn.Module):
    def __init__(self):
        super(MLP5, self).__init__()
        self.fc1 = nn.Linear(3, 31)
        self.fc2 = nn.Linear(31, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 31)
        self.fc5 = nn.Linear(31, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# class MLP5(nn.Module):
#     def __init__(self):
#         super(MLP5, self).__init__()
#         self.fc1 = nn.Linear(3, 79)
#         self.fc2 = nn.Linear(79, 79)
#         self.fc3 = nn.Linear(79, 79)
#         self.fc4 = nn.Linear(79, 36)
#         self.fc5 = nn.Linear(36, 3)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = self.fc5(x)
#         return x


class MLP7(nn.Module):
    def __init__(self):
        super(MLP7, self).__init__()
        self.fc1 = nn.Linear(3, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 25)
        self.fc4 = nn.Linear(25, 25)
        self.fc5 = nn.Linear(25, 24)
        self.fc6 = nn.Linear(24, 24)
        self.fc7 = nn.Linear(24, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x


# class MLP7(nn.Module):
#     def __init__(self):
#         super(MLP7, self).__init__()
#         self.fc1 = nn.Linear(3, 79)
#         self.fc2 = nn.Linear(79, 79)
#         self.fc3 = nn.Linear(79, 79)
#         self.fc4 = nn.Linear(79, 79)
#         self.fc5 = nn.Linear(79, 79)
#         self.fc6 = nn.Linear(79, 36)
#         self.fc7 = nn.Linear(36, 3)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = F.relu(self.fc5(x))
#         x = F.relu(self.fc6(x))
#         x = self.fc7(x)
#         return x


class MLP9(nn.Module):
    def __init__(self):
        super(MLP9, self).__init__()
        self.fc1 = nn.Linear(3, 21)
        self.fc2 = nn.Linear(21, 21)
        self.fc3 = nn.Linear(21, 21)
        self.fc4 = nn.Linear(21, 21)
        self.fc5 = nn.Linear(21, 21)
        self.fc6 = nn.Linear(21, 21)
        self.fc7 = nn.Linear(21, 21)
        self.fc8 = nn.Linear(21, 20)
        self.fc9 = nn.Linear(20, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x


# class MLP9(nn.Module):
#     def __init__(self):
#         super(MLP9, self).__init__()
#         self.fc1 = nn.Linear(3, 79)
#         self.fc2 = nn.Linear(79, 79)
#         self.fc3 = nn.Linear(79, 79)
#         self.fc4 = nn.Linear(79, 79)
#         self.fc5 = nn.Linear(79, 79)
#         self.fc6 = nn.Linear(79, 79)
#         self.fc7 = nn.Linear(79, 79)
#         self.fc8 = nn.Linear(79, 36)
#         self.fc9 = nn.Linear(36, 3)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = F.relu(self.fc5(x))
#         x = F.relu(self.fc6(x))
#         x = F.relu(self.fc7(x))
#         x = F.relu(self.fc8(x))
#         x = self.fc9(x)
#         return x


class Generator(nn.Module):
    def __init__(self, latent_dim, spectrum_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 20),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(20, 40),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(40, 80),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(80, 160),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(160, spectrum_dim),
        )

    def forward(self, z):
        spectrum = self.model(z)
        return spectrum


class Discriminator(nn.Module):
    def __init__(self, spectrum_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(spectrum_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, spectrum):
        score = self.model(spectrum)
        return score