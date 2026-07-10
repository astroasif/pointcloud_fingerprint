import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class PointNet2Cls(nn.Module):
    def __init__(self, point_size, latent_size, num_classes, normal_channel=False):
        super(PointNet2Cls, self).__init__()
        self.latent_size = latent_size
        self.point_size = point_size
        self.normal_channel = normal_channel

        in_channel = 3 if normal_channel else 0
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.out = nn.Linear(256, num_classes)

    def encoder(self, x):
        B, _, _ = x.shape
        norm = x[:, 3:, :] if self.normal_channel else None
        x = x[:, :3, :]
        l1_xyz, l1_points = self.sa1(x, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return x

    def forward(self, x):
        feats = self.encoder(x)         # [B, 256]
        logits = self.out(feats)        # [B, C]
        return logits



class PointNet2Cls_51(nn.Module):
    def __init__(self, point_size, latent_size, num_classes, normal_channel=False):
        super(PointNet2Cls_51, self).__init__()
        self.latent_size = latent_size
        self.point_size = point_size
        self.normal_channel = normal_channel

        in_channel = 3 if normal_channel else 0
        self.sa1 = PointNetSetAbstractionMsg(
            512, [0.06, 0.10, 0.16], [24, 48, 96], in_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.12, 0.20, 0.32], [48, 96, 160], 320,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.out = nn.Linear(256, num_classes)

    def encoder(self, x):
        B, _, _ = x.shape
        norm = x[:, 3:, :] if self.normal_channel else None
        x = x[:, :3, :]
        l1_xyz, l1_points = self.sa1(x, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return x

    def forward(self, x):
        feats = self.encoder(x)
        return self.out(feats)


class PointNet2Cls_59(nn.Module):
    def __init__(self, point_size, latent_size, num_classes, normal_channel=False):
        super(PointNet2Cls_59, self).__init__()
        self.latent_size = latent_size
        self.point_size = point_size
        self.normal_channel = normal_channel

        in_channel = 3 if normal_channel else 0
        self.sa1 = PointNetSetAbstractionMsg(
            1024, [0.06, 0.10, 0.16], [24, 48, 96], in_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            256, [0.12, 0.20, 0.32], [48, 96, 160], 320,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.out = nn.Linear(256, num_classes)

    def encoder(self, x):
        B, _, _ = x.shape
        norm = x[:, 3:, :] if self.normal_channel else None
        x = x[:, :3, :]
        l1_xyz, l1_points = self.sa1(x, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return x

    def forward(self, x):
        feats = self.encoder(x)
        return self.out(feats)




class PointCloudAE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAE, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)
        
        self.dec1 = nn.Linear(self.latent_size,2048)
        self.dec2 = nn.Linear(2048,2048)
        self.dec3 = nn.Linear(2048,self.point_size*3)

    def encoder(self, x):
        x = x[:, :3, :] 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x
    
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)
    
    def forward(self, x):
        x = self.encoder(x)
        global_feature = x
        x = self.decoder(x)

        return x, global_feature
    

class PointNet2AE(nn.Module):
    def __init__(self, point_size, latent_size, normal_channel=False):
        super(PointNet2AE, self).__init__()

        self.latent_size = latent_size
        self.point_size = point_size
        self.normal_channel = normal_channel
        self.point_dims = 6 if normal_channel else 3

        in_channel = 3 if normal_channel else 0
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)

        self.bn3 = nn.BatchNorm1d(self.latent_size)
        
        self.dec1 = nn.Linear(self.latent_size,1024)
        self.dec2 = nn.Linear(1024,1024)
        self.dec3 = nn.Linear(1024, self.point_size*self.point_dims)

    def encoder(self, x): 
        B, _, _ = x.shape
        if self.normal_channel:
            norm = x[:, 3:, :]
            x = x[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(x, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.bn1(self.fc1(x))
        x = x.view(-1, self.latent_size)
        return x
    
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, self.point_dims)
    
    def forward(self, x):
        x = self.encoder(x)
        return x


class PointNet2Reg(nn.Module):
    def __init__(self, point_size, latent_size, normal_channel=False):
        super(PointNet2Reg, self).__init__()

        self.latent_size = latent_size
        self.point_size = point_size
        self.normal_channel = normal_channel
        self.point_dims = 6 if normal_channel else 3

        in_channel = 3 if normal_channel else 0
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)

        self.bn3 = nn.BatchNorm1d(self.latent_size)
        
        self.dec1 = nn.Linear(self.latent_size,256)
        self.dec2 = nn.Linear(256, 128)
        self.dec3 = nn.Linear(128, 16)
        self.dec4 = nn.Linear(16, 1)

    def encoder(self, x): 
        B, _, _ = x.shape
        if self.normal_channel:
            norm = x[:, 3:, :]
            x = x[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(x, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.bn1(self.fc1(x))
        x = x.view(-1, self.latent_size)
        return x
    
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = self.dec4(x)
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        global_feature = x
        x = self.decoder(x)
        return x, global_feature
    
