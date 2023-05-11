import torch
import torchvision.models as torchmodels
import torch.nn as nn
import torch.nn.functional as F


class VGGSSL(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task
        
        self.encoder = torchmodels.vgg16()
        # self.model = torchmodels.vgg16(weights='DEFAULT')        
        self.encoder.classifier = torch.nn.Sequential(*list(self.encoder.classifier.children())[:-3])
        if self.task == "RP":
            self.classifier = nn.Linear(4096, 2)
        elif self.task == "TS":
            self.classifier = nn.Linear(4096*2, 2)
        elif self.task == "CPC":
            self.gAR = nn.GRU(4096, 100) # hidden size = 100, per (Banville et al, 2020) experiment
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x):
        if self.task == "CPC":
            # x: N x 3 (context, future, negative) x samples x R x G x B
            embeds = []
            for n in range(x.shape[0]): # for each batch sample
                tup = []
                for samples in x[n]:
                    tup.append([self.encoder(sample) for sample in samples])
                embeds.append(tup)
            
            # embeds: N x 3 x samples x 4096 (samples are different between context, future, and negative
            context = self.gAR(embeds[:,0,:,:]) # N x 100
            
            z = [(context[n], embeds[n,1,:,:], embeds[n,2,:,:]) for n in range(len(embeds))]
        else:
            # If task == RP, embeds is a list/tuple of two embeddings
            #    task == TS, embeds is a list/tuple of three embeddings
            embeds = [self.encoder(x[:,i,:,:,:]) for i in range(x.shape[1])] # x: N (Batch_size) x Sample_size x R x G x B
            if self.task == "RP":
                g = torch.abs(embeds[0] - embeds[1])
            elif self.task == "TS":
                g = torch.cat([torch.abs(embeds[0] - embeds[1]), torch.abs(embeds[1] - embeds[2])], dim=1)
            z = self.classifier(g)
        
            del g
            del embeds
        return z
    
    def train(self, num_epochs, print_every, dataloader, model_save_dir, optimizer, writer):
        self.encoder.train()
        self.classifier.train()
        for e in range(num_epochs):
            for t, (sample, label) in enumerate(dataloader):
                label = label.to(device=self.device, dtype=torch.long)
                sample = sample.to(device=self.device)
                logit = self(sample)
                
                optimizer.zero_grad()
                loss =  F.cross_entropy(logit, label)
                loss.backward()
                optimizer.step()

                if t % print_every == 0:
                    writer.add_scalar("Loss/train", loss.item(), e*len(dataloader)+t)
                    print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))

                del label
                del logit
                del loss

            # Save model every print_every epochs
            if e > 0 and e % print_every == 0:
                torch.save(self.state_dict(), f"{model_save_dir}/epoch_{e}")

    
