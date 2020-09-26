# torch import
import torch
import torch.nn as nn

# general import
import matplotlib.pyplot as plt

# FGSM attack
class FGSMAttack(object):
    def __init__(self, model, epsilons, test_dataloader, device, target=None):
        self.model = model
        self.epsilons = epsilons
        self.test_dataloader = test_dataloader
        self.device = device
        self.target = target
        self.adv_examples = {}
        
    def perturb(self, x, eps, grad):
        x_prime = None
        if self.target:
            x_prime = x - eps * grad.sign()
        else:
            x_prime = x + eps * grad.sign()
            
        # keep image data in the [0,1] range
        x_prime = torch.clamp(x_prime, 0, 1)
        return x_prime
    
    def run(self):
        # run the attack for each epsilon
        for epsReal in self.epsilons:
            self.adv_examples[epsReal] = [] # store some adv samples for visualization
            eps = epsReal - 1e-7 # small constant to offset floating-point errors
            successful_attacks = 0

            for data, label in self.test_dataloader:
                # send dat to device
                data, label = data.to(self.device), label.to(self.device)
                
                # FGSM attack requires gradients w.r.t. the data
                data.requires_grad = True
                
                output = self.model(data)
                init_pred = output.argmax(dim=1, keepdim=True)
                if init_pred.item() != label.item():
                    # image is not correctly predicted to begin with, skip
                    continue
                if self.target and self.target == label.item():
                    # if the image has the target class, skip
                    continue
                    
                # calculate the loss
                L = nn.CrossEntropyLoss()
                loss = None
                if self.target:
                    # in a target attack, we take the loss w.r.t. the target label
                    loss = L(output, torch.tensor([self.target], dtype=torch.long))
                else:
                    loss = L(output, torch.tensor([init_pred.item()], dtype=torch.long))
                
                # zero out all existing gradients
                self.model.zero_grad()
                # calculate gradients
                loss.backward()
                data_grad = data.grad
                
                perturbed_data = self.perturb(data, eps, data_grad)
                
                # predict class for adversarial sample
                adv_output = self.model(perturbed_data)
                adv_pred = adv_output.argmax(dim=1, keepdim=True)
                
                if self.target:
                    if adv_pred.item() == self.target:
                        successful_attacks += 1
                        if len(self.adv_examples[epsReal]) < 5:
                            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                            self.adv_examples[epsReal].append( (init_pred.item(), adv_pred.item(), adv_ex ) )       
                else:
                    if adv_pred.item() != init_pred.item():
                        successful_attacks += 1
                        if len(self.adv_examples[epsReal]) < 5:
                            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                            self.adv_examples[epsReal].append( (init_pred.item(), adv_pred.item(), adv_ex ) )
                
            # print status line
            success_rate = successful_attacks / float(len(self.test_dataloader))
            print("Epsilon: {}\tAttack Success Rate = {} / {} = {}".format(epsReal, successful_attacks, len(self.test_dataloader), success_rate))
            
    def visualize(self):
        plt.figure(figsize=(8,10))
        cnt = 0
        for eps, adv_examples in self.adv_examples.items():
            for index, data in enumerate(adv_examples):
                cnt += 1
                plt.subplot(len(self.adv_examples.keys()), len(adv_examples), cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if index == 0:
                    plt.ylabel("Eps: {}".format(eps), fontsize=14)
                
                orig, adv, adv_ex = data
                plt.title("{} -> {}".format(orig, adv))
                plt.imshow(adv_ex, cmap="gray")
            # round cnt up to next multiple of 5
            cnt += 4
            cnt -= cnt % 5
        plt.tight_layout()
        if self.target:
            plt.savefig("data/img/t{}_fgsm.png".format(self.target))
        else:
            plt.savefig("data/img/ut_fgsm.png")
        plt.show()