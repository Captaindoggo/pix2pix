import torch
import torch.nn as nn
from torch.optim import Adam



class C_k(nn.Module):
    def __init__(self, n_inn, n_filters, stride=2):
        super(C_k, self).__init__()
        if n_inn == 3 or n_inn == 6:
            self.conv_br = nn.Sequential(
                nn.Conv2d(n_inn, n_filters, 4, stride=stride, padding=1),
                nn.LeakyReLU(0.2, True)
            )
        else:
            self.conv_br = nn.Sequential(
                nn.Conv2d(n_inn, n_filters, 4, stride=stride, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.LeakyReLU(0.2, True)
            )

    def forward(self, x):
        return self.conv_br(x)


class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        dims = [[3, 64], [64, 128], [128, 256], [256, 512], [512, 512], [512, 512], [512, 512], [512, 512]]
        self.convs = nn.ModuleList([C_k(i[0], i[1]) for i in dims])
        self.device = device

    def forward(self, x):
        skips = []
        for layer in self.convs:
            x = layer(x)
            skips.append(x)

        return skips


class CD_k(nn.Module):
    def __init__(self, n_inn, n_filters, dropout=False, stride=2):
        super(CD_k, self).__init__()
        if dropout:
            self.conv_bdr = nn.Sequential(
                nn.ConvTranspose2d(n_inn, n_filters, 4, stride=stride, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.Dropout(0.5),
                nn.ReLU(True)
            )
        else:
            self.conv_bdr = nn.Sequential(
                nn.ConvTranspose2d(n_inn, n_filters, 4, stride=stride, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(True)
            )

    def forward(self, x):
        return self.conv_bdr(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        dims = [[512, 512, True], [1024, 512, True], [1024, 512, True], [1024, 512, False], [1024, 512, False],
                [768, 512, False], [640, 256, False], [320, 128, False]]
        self.convs = nn.ModuleList([CD_k(i[0], i[1]) for i in dims])
        self.last = nn.ConvTranspose2d(128, 3, 1, 1)
        self.t = nn.Tanh()

    def forward(self, skips):
        skips.reverse()
        x = skips[0]
        for i, layer in enumerate(self.convs):
            if i != 0:
                x = torch.cat((x, skips[i]), dim=1)
            x = layer(x)
        x = self.t(self.last(x))
        return x


class Generator(nn.Module):
    def __init__(self, device):
        super(Generator, self).__init__()

        self.encoder = Encoder(device)

        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        dims = [[6, 64, 2], [64, 128, 2], [128, 256, 2], [256, 512, 1]]
        self.convs = nn.ModuleList([C_k(i[0], i[1], stride=i[2]) for i in dims])
        self.last = nn.Conv2d(dims[-1][1], 1, 1, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = self.last(x)
        # x = self.sigm(x)
        return x

    def require_grad(self, req_g):
        for p in self.parameters():
            p.requires_grad = req_g


class Pix2Pix(nn.Module):
    def __init__(self, device, lmbda=100):
        super(Pix2Pix, self).__init__()

        self.generator = Generator(device)
        self.discriminator = Discriminator()
        self.g_optimizer = Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        self.lmbda = lmbda
        self.device = device

    def forward(self, A):
        gen_B = self.generator(A)
        return gen_B

    def D_backward(self, A_B, A_gen_B):
        pred = self.discriminator(A_gen_B.detach())
        gen_loss = self.criterion_gan(pred, torch.zeros(pred.shape).to(self.device))

        pred = self.discriminator(A_B)
        truth_loss = self.criterion_gan(pred, torch.ones(pred.shape).to(self.device))

        final_D_loss = (gen_loss + truth_loss) / 2.0
        batch_loss = final_D_loss.item()
        final_D_loss.backward()
        return batch_loss

    def G_backward(self, B, gen_B, A_gen_B):
        pred = self.discriminator(A_gen_B)
        bce_loss = self.criterion_gan(pred, torch.ones(pred.shape).to(self.device))

        l1_loss = self.criterion_l1(gen_B, B) * self.lmbda
        final_G_loss = bce_loss + l1_loss
        batch_loss = final_G_loss.item()
        final_G_loss.backward()

        return batch_loss

    def step(self, A, B):
        gen_B = self.generator(A)

        A_gen_B = torch.cat((A, gen_B), 1)
        A_B = torch.cat((A, B), 1)

        self.discriminator.require_grad(True)
        self.d_optimizer.zero_grad()
        dloss = self.D_backward(A_B, A_gen_B)
        self.d_optimizer.step()

        self.discriminator.require_grad(False)

        self.g_optimizer.zero_grad()
        gloss = self.G_backward(B, gen_B, A_gen_B)
        self.g_optimizer.step()
        return gloss, dloss