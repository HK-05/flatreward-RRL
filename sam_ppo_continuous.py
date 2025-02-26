import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
import numpy as np
from torch.distributions import Beta, Normal
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)


    def step(self, closure):
        assert closure is not None, "SAM optimizer requires a closure that reevaluates the model and returns the loss"

        loss = closure()
        loss.backward()
        grad_norm = self._grad_norm()
        scale = self.param_groups[0]['rho'] / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.data.add_(e_w) 
                
        loss = closure()
        loss.backward()

        self.base_optimizer.step()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.data.sub_(e_w)  

        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.detach().norm(p=2).to(device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ])
        )
        return norm

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)
        return mean

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device='cpu'):
        self.to(device)
        self.load_state_dict(torch.load(filename, map_location=torch.device(device)))

class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.full((1, args.action_dim), -0.5)) 
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std).clamp(min=1e-3, max=10)  
        dist = Normal(mean, std)
        return dist

    def compute_actor_loss(self, s, a, old_logprob, adv, epsilon, entropy_coef):
        dist = self.get_dist(s)
        logprob_now = dist.log_prob(a).sum(dim=1, keepdim=True)
        entropy = dist.entropy().sum(dim=1, keepdim=True)
        ratios = torch.exp(logprob_now - old_logprob)

        surr1 = ratios * adv
        surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * adv
        actor_loss = -torch.min(surr1, surr2) - entropy_coef * entropy
        return actor_loss.mean()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device='cpu'):
        self.to(device)
        self.load_state_dict(torch.load(filename, map_location=torch.device(device)))
    
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s

    def compute_critic_loss(self, s, v_target):
        v_pred = self.forward(s)
        critic_loss = F.mse_loss(v_pred, v_target)
        return critic_loss.mean()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device='cpu'):
        self.to(device)
        self.load_state_dict(torch.load(filename, map_location=torch.device(device)))

class PPO_continuous():
    def __init__(self, args):
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient

        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.adaptive_alpha = args.adaptive_alpha
        self.weight_reg = args.weight_reg

        # adaptive_alpha 
        if self.adaptive_alpha:
            self.target_entropy = -args.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_a)
        else:
            self.alpha = 0.0

        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args)
        else:
            self.actor = Actor_Gaussian(args)
        self.critic = Critic(args)

        self.use_sam = args.use_sam 
        self.rho = args.rho         

        if self.use_sam:
            base_optimizer_actor = Adam
            base_optimizer_critic = Adam

            self.optimizer_actor = SAM(
                self.actor.parameters(),
                base_optimizer_actor,
                rho=self.rho,
                lr=self.lr_a,
                eps=1e-5 if self.set_adam_eps else 1e-8
            )
            self.optimizer_critic = SAM(
                self.critic.parameters(),
                base_optimizer_critic,
                rho=self.rho,
                lr=self.lr_c,
                eps=1e-5 if self.set_adam_eps else 1e-8
            )
        else:
            if self.set_adam_eps:
                self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
                self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
            else:
                self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
                self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.writer = SummaryWriter(log_dir='runs/SAM_PPO_continuous')

        print(f"Use SAM: {self.use_sam}, rho: {self.rho if hasattr(args, 'rho') else 'N/A'}")
        # print(f"Actor optimizer type: {type(self.optimizer_actor)}")
        # print(f"Critic optimizer type: {type(self.optimizer_critic)}")

    def evaluate(self, s): 
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            if self.policy_dist == "Beta":
                a = self.actor.mean(s).detach().numpy().flatten()
            else:
                a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  
                a_logprob = dist.log_prob(a).sum(dim=1, keepdim=True)
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  
                a = torch.clamp(a, -self.max_action, self.max_action)  
                a_logprob = dist.log_prob(a).sum(dim=1, keepdim=True)
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    def update(self, replay_buffer, total_steps):
        # print(f"Updating with use_sam: {self.use_sam}")
        # print(f"Actor optimizer type: {type(self.optimizer_actor)}")
        # print(f"Critic optimizer type: {type(self.optimizer_critic)}")

        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # 학습 데이터 가져오기

        adv = []
        gae = 0
        with torch.no_grad():  
            vs = self.critic(s)
            vs_ = self.critic(s_)
            # IPM uncertainty set
            reg_norm, weight_norm, bias_norm = 0, [], []
            for layer in self.critic.children():
                if isinstance(layer, nn.Linear):
                    weight_norm.append(torch.norm(layer.state_dict()['weight']) ** 2)
                    bias_norm.append(torch.norm(layer.state_dict()['bias']) ** 2)
            reg_norm = torch.sqrt(torch.sum(torch.stack(weight_norm)) + torch.sum(torch.stack(bias_norm[0:-1])))
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs - self.alpha * a_logprob.sum(dim=1, keepdim=True) - self.weight_reg * reg_norm
            for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(s.device)
            v_target = adv + vs + self.alpha * a_logprob.sum(dim=1, keepdim=True)
            if self.use_adv_norm:  
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))


        for epoch in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):

          
                def closure_actor():
                    return self.actor.compute_actor_loss(
                        s[index],
                        a[index],
                        a_logprob[index],
                        adv[index],
                        self.epsilon,
                        self.entropy_coef
                    )

                if self.use_sam:
                    
                    self.optimizer_actor.step(closure_actor)
                else:
                    
                    actor_loss = self.actor.compute_actor_loss(
                        s[index],
                        a[index],
                        a_logprob[index],
                        adv[index],
                        self.epsilon,
                        self.entropy_coef
                    )
                    self.optimizer_actor.zero_grad()
                    actor_loss.backward()
                    if self.use_grad_clip: 
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.optimizer_actor.step()

              
                def closure_critic():
                    return self.critic.compute_critic_loss(s[index], v_target[index])

                if self.use_sam:
               
                    self.optimizer_critic.step(closure_critic)
                else:
              
                    critic_loss = self.critic.compute_critic_loss(s[index], v_target[index])
                    self.optimizer_critic.zero_grad()
                    critic_loss.backward()
                    if self.use_grad_clip: 
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.optimizer_critic.step()

               
                with torch.no_grad():
                    if self.policy_dist == "Gaussian":
                        self.actor.log_std.data.clamp_(-20, 2)
                    self.actor.fc1.weight.data.clamp_(-5, 5)
                    self.actor.fc2.weight.data.clamp_(-5, 5)
                    self.actor.mean_layer.weight.data.clamp_(-5, 5)
                    self.critic.fc1.weight.data.clamp_(-5, 5)
                    self.critic.fc2.weight.data.clamp_(-5, 5)
                    self.critic.fc3.weight.data.clamp_(-5, 5)

              
                if epoch % 10 == 0 and index == 0:
                    if self.policy_dist == "Gaussian":
                        log_std_val = self.actor.log_std.mean().item()
                        self.writer.add_scalar('log_std', log_std_val, total_steps)
                    if self.use_sam:
                      
                        actor_loss_val = closure_actor().item()
                        critic_loss_val = closure_critic().item()
                    else:
                        actor_loss_val = actor_loss.item()
                        critic_loss_val = critic_loss.item()
                    self.writer.add_scalar('Actor Loss', actor_loss_val, total_steps)
                    self.writer.add_scalar('Critic Loss', critic_loss_val, total_steps)
                    print(f"Epoch: {epoch}, Actor Loss: {actor_loss_val}, Critic Loss: {critic_loss_val}")

                  
                    if torch.isnan(torch.tensor(actor_loss_val)) or torch.isnan(torch.tensor(critic_loss_val)):
                        print("NaN detected in actor or critic loss")
                        raise ValueError("NaN detected in loss")

        if self.use_lr_decay:  
            self.lr_decay(total_steps)

        if self.adaptive_alpha:
            alpha_loss = -(self.log_alpha.exp() * (a_logprob.sum(dim=1, keepdim=True) + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        if self.use_sam and hasattr(self.optimizer_actor, 'base_optimizer'):
            for p in self.optimizer_actor.base_optimizer.param_groups:
                p['lr'] = lr_a_now
            for p in self.optimizer_critic.base_optimizer.param_groups:
                p['lr'] = lr_c_now
        else:
            for p in self.optimizer_actor.param_groups:
                p['lr'] = lr_a_now
            for p in self.optimizer_critic.param_groups:
                p['lr'] = lr_c_now
