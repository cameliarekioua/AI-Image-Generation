import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.emb_layer = nn.Linear(emb_dim, in_ch)
    def forward(self, x, cond):
        emb = self.emb_layer(cond)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        x = x+emb
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNetConditionnel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, emb_dim=256):
        super().__init__()
        self.emb_dim = emb_dim

        # Embedding temps + label (fusion + projection)
        self.time_embed = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.label_embed = nn.Linear(512, emb_dim)

        # On concatene le conditionnement comme un canal ajouté à l'image
        #dcp input channels = in_channels + 1
        self.down1 = DoubleConv(in_channels, 64)
        self.pool = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.middle = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(512 + 512, 512)  # concat avec down4
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(256 + 256, 256)  # concat avec down3
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(128 + 128, 128)     # concat avec down2
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv4 = DoubleConv(64 + 64, 64)     # concat avec down1

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x, t, labels):
        B, C, H, W = x.shape
        t_emb = self.time_embed(t)       
        lbl_emb = self.label_embed(labels)  
        cond = t_emb + lbl_emb            # (B, emb_dim)
        #print(H,W)

    
  
        d1 = self.down1(x,cond)   
        p1 = self.pool(d1)        
        d2 = self.down2(p1, cond)      
        p2 = self.pool(d2)    
        d3 = self.down3(p2, cond)    
        p3 = self.pool(d3)       
        d4 = self.down4(p3, cond)  
        p4 = self.pool(d4)       

        m = self.middle(p4, cond)       

        #ajout skip connections
        u1 = self.up1(m)        
        u1 = torch.cat([u1, d4], dim=1) 
        u1 = self.upconv1(u1, cond)  

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)  
        u2 = self.upconv2(u2, cond)   

        u3 = self.up3(u2)         
        u3 = torch.cat([u3, d2], dim=1) 
        u3 = self.upconv3(u3, cond)    

        u4 = self.up4(u3)     
        u4 = torch.cat([u4, d1], dim=1) 
        u4 = self.upconv4(u4, cond)    

        out = self.out(u4)        
        return out
