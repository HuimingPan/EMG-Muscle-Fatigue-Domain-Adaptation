import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
ALPHA = 10
BETA = 0.75
GAMMA = 100.
class BaseTrainer:

    def __init__(self, subject_num, config, Estimator, DomainClassifier, Dataset, **kwargs):
        self.subject_num = subject_num
        self.config = config
        self.setup_info_holders()
        self.setup_labels()
        self.setup_data_loaders(Dataset, **kwargs)
        self.setup_networks(Estimator, DomainClassifier)
        self.setup_optimizers()

    def setup_info_holders(self, **kwargs):
        self.metrics_during_training = pd.DataFrame(
            columns=["train_loss", "train_classify_loss", "train_estimate_loss", "test_nf_loss",
                     "test_nf_classify_loss", "test_nf_estimate_loss", "test_nf_acc", "test_f_loss",
                     "test_f_classify_loss", "test_f_estimate_loss", "test_f_acc"])
        self.source_best_rmse = 100
        self.target_best_rmse = 100
        self.source_final_rmse = None
        self.target_final_rmse = None
        self.best_source_net_state = None
        self.best_target_net_state = None
        self.source_test_losses = []
        self.target_test_losses = []
        self.source_test_rmse = []
        self.target_test_rmse = []
        self.source_test_acc = []
        self.target_test_acc = []
        self.eval_category = None
        self.true_category = None
        self.source_eval_estimate = []
        self.target_eval_estimate = []
        self.source_eval_label = []
        self.target_eval_label = []
        self.iters = 0

    def setup_labels(self, **kwargs):
        self.source_disc_labels = torch.zeros(self.config.BATCH_SIZE).requires_grad_(False)
        self.target_disc_labels = torch.ones(self.config.BATCH_SIZE).requires_grad_(False)

    def setup_data_loaders(self, Dataset, test_trials=None, **kwargs):
        if test_trials is None:
            test_trials = [3, 6, 9]
        source_test_trials = [f"{self.subject_num}-1-{i}" for i in test_trials]
        target_test_trials = [f"{self.subject_num}-2-{i}" for i in test_trials]
        train_trials = [i for i in range(1, 10) if i not in test_trials]
        source_train_trials = [f"{self.subject_num}-1-{i}" for i in train_trials]
        target_train_trials = [f"{self.subject_num}-2-{i}" for i in train_trials]

        self.source_train_loader = DataLoader(Dataset(self.config.update(TRIALS=source_train_trials)),
                                              batch_size=self.config.BATCH_SIZE, shuffle=True)
        self.target_train_loader = DataLoader(Dataset(self.config.update(TRIALS=target_train_trials)),
                                              batch_size=self.config.BATCH_SIZE, shuffle=True)
        self.source_test_loader = DataLoader(Dataset(self.config.update(TRIALS=source_test_trials)),
                                             batch_size=self.config.BATCH_SIZE, shuffle=False)
        self.target_test_loader = DataLoader(Dataset(self.config.update(TRIALS=target_test_trials)),
                                             batch_size=self.config.BATCH_SIZE, shuffle=False)
        self.n_batch = min(len(self.source_train_loader), len(self.target_train_loader))

    def setup_networks(self, Estimator, DomainClassifier, **kwargs):
        self.net = Estimator(input_dim=3*len(self.config.ELECTRODES))
        self.encoder = self.net.encode
        self.discriminitor = DomainClassifier(self.encoder)
        if self.config.CUDA:
            self.net.cuda()
            self.discriminitor.cuda()

    def setup_optimizers(self):
        self.optimizer_net = torch.optim.Adam(self.net.parameters(), lr=self.config.LR)
        self.optimizer_encoder = torch.optim.Adam(self.net.parameters(), lr=self.config.LR)
        self.optimizer_discriminator = torch.optim.Adam(self.discriminitor.parameters(), lr=self.config.LR)
        self.loss_fn_classify = nn.BCELoss()
        self.loss_fn_estimate = nn.MSELoss()

    def train_loop(self):
        self.net.train()
        tbar = tqdm(enumerate(zip(self.source_train_loader, self.target_train_loader)))
        net_loss = 0.0
        disc_loss = 0.0
        total_loss = 0.0
        for i, ((source_data, source_label, _), (target_data, _, _)) in tbar:
            ##############################
            # update learning parameters #
            ##############################
            self.iters += 1
            p = self.iters / (self.config.EPOCHS * self.config.BATCH_SIZE)

            lambd = (2 / (1 + np.exp(-GAMMA * p))) - 1
            self.discriminitor.update_lambd(lambd)
            lr = self.config.LR / (1 + 10 * ALPHA * p) ** BETA
            self.optimizer_net.lr = lr
            self.optimizer_encoder.lr = lr
            self.optimizer_discriminator.lr = lr

            #########################################################################
            # set batch size in cases where source and target domain differ in size #
            #########################################################################
            curr_batch_size = min(source_data.shape[0], target_data.shape[0])
            source_data = source_data[:curr_batch_size]
            source_label = source_label[:curr_batch_size]
            target_data = target_data[:curr_batch_size]
            source_disc_labels = self.source_disc_labels[:curr_batch_size]
            target_disc_labels = self.target_disc_labels[:curr_batch_size]

            if self.config.CUDA:
                source_data = source_data.cuda()
                source_label = source_label.cuda()
                target_data = target_data.cuda()

                source_disc_labels = source_disc_labels.cuda()
                target_disc_labels = target_disc_labels.cuda()

            #####################
            #   Train network   #
            #####################
            self.optimizer_net.zero_grad()
            net_output = self.net(source_data)
            estimate_loss = F.mse_loss(net_output, source_label)
            estimate_loss.backward()
            self.optimizer_net.step()
            net_loss += estimate_loss.item()

            #########################################
            # Train encoder on Source + Target data #
            #########################################
            self.optimizer_encoder.zero_grad()
            self.optimizer_discriminator.zero_grad()
            disc_input = torch.cat((source_data, target_data), 0)
            disc_output = self.discriminitor(disc_input)
            disc_labels = torch.cat((source_disc_labels, target_disc_labels), 0)
            disc_loss = self.loss_fn_classify(disc_output, disc_labels)

            disc_loss.backward()
            self.optimizer_discriminator.step()
            self.optimizer_encoder.step()

            disc_loss += disc_loss.item()
            total_loss += estimate_loss.item() - lambd * disc_loss.item()

            ####################################
            # Update progress bar description  #
            ####################################
            description = (f'Net Loss: {net_loss / (i + 1):.4f}; '
                           f'Discriminator Loss: {disc_loss / (i + 1):.4f}; '
                           f'Total Loss: {total_loss / (i + 1):.4f}')
            tbar.set_description(description)

    def test_loop(self):
        self.net.eval()
        self.discriminitor.eval()
        ####################
        # Test Source Data #
        ####################
        test_loss = 0
        source_test_classify_output = []
        with torch.no_grad():
            for data, labels, _ in self.source_test_loader:
                if self.config.CUDA:
                    data = data.cuda()
                    labels = labels.cuda()
                output = self.net(data)
                test_loss += F.mse_loss(output, labels).item() * len(labels)
                classify_output = (self.discriminitor(data) > 0.5).to(dtype=torch.int, device="cpu")
                source_test_classify_output.append(classify_output)
        source_test_classify_output = torch.cat(source_test_classify_output, 0)
        acc = np.mean(source_test_classify_output == np.zeros_like(source_test_classify_output)).item()
        self.source_test_acc.append(acc)

        test_loss /= len(self.source_test_loader.dataset)
        rmse_source = np.sqrt(test_loss)
        self.source_test_losses.append(test_loss)
        self.source_test_rmse.append(rmse_source)

        if rmse_source < self.source_best_rmse:
            self.source_best_rmse = rmse_source
            self.best_source_net_state = self.net.state_dict()

        ####################
        # Test Target Data #
        ####################
        test_loss = 0
        target_test_classify_output = []
        with torch.no_grad():
            for data, labels, _ in self.target_test_loader:
                if self.config.CUDA:
                    data = data.cuda()
                    labels = labels.cuda()
                output = self.net(data)
                test_loss += F.mse_loss(output, labels).item() * len(labels)
                classify_output = (self.discriminitor(data) > 0.5).to(dtype=torch.int, device="cpu")
                target_test_classify_output.append(classify_output)
        target_test_classify_output = torch.cat(target_test_classify_output, 0)
        acc = np.mean(target_test_classify_output == np.ones_like(target_test_classify_output)).item()

        test_loss /= len(self.target_test_loader.dataset)
        rmse_target = np.sqrt(test_loss)
        self.target_test_losses.append(test_loss)
        self.target_test_rmse.append(rmse_target)

        if rmse_target < self.target_best_rmse:
            self.target_best_rmse = rmse_target
            self.best_target_net_state = self.net.state_dict()

    def train(self):
        for i in range(self.config.EPOCHS):
            print(f"====================Epoch {i}====================")
            self.train_loop()
            self.test_loop()

    def evaluation(self, model="final", **kwargs):
        suffix = kwargs.get("suffix", "")

        ############################
        # Evaluate the final model #
        ############################
        source_eval_estimate = []
        source_eval_category = []
        source_eval_label = []
        target_eval_estimate = []
        target_eval_category = []
        target_eval_label = []
        if model == "final":
            #######################################
            # Evaluate Model on Source and Target #
            #######################################
            for X, y, _ in self.source_test_loader:
                if self.config.CUDA:
                    X = X.cuda()
                    y = y.cuda()
                estimate = self.net(X).cpu().detach()
                category = self.discriminitor(X).cpu().detach()

                source_eval_estimate.append(estimate)
                source_eval_label.append(y.cpu().detach())
                source_eval_category.append((category < 0.5).to(torch.int))

            for X, y, _ in self.target_test_loader:
                if self.config.CUDA:
                    X = X.cuda()
                    y = y.cuda()
                estimate = self.net(X).cpu().detach()
                category = self.discriminitor(X).cpu().detach()

                target_eval_estimate.append(estimate)
                target_eval_label.append(y.cpu().detach())
                target_eval_category.append((category > 0.5).to(torch.int))

            ###########################
            # Concatenate the results #
            ###########################
            self.source_eval_estimate = torch.cat(source_eval_estimate, 0)
            self.source_eval_label = torch.cat(source_eval_label, 0)
            self.source_eval_category = torch.cat(source_eval_category, 0)
            self.target_eval_estimate = torch.cat(target_eval_estimate, 0)
            self.target_eval_label = torch.cat(target_eval_label, 0)
            self.target_eval_category = torch.cat(target_eval_category, 0)

            self.eval_category = torch.cat((self.source_eval_category, self.target_eval_category), 0).squeeze()
            self.true_category = torch.cat(
                (torch.zeros_like(self.source_eval_category), torch.ones_like(self.target_eval_category)), 0).squeeze()

            self.source_final_rmse = torch.sqrt(F.mse_loss(self.source_eval_estimate, self.source_eval_label))
            self.target_final_rmse = torch.sqrt(F.mse_loss(self.target_eval_estimate, self.target_eval_label))

            mini_size = min(self.source_eval_estimate.shape[0], self.target_eval_estimate.shape[0])
            df = pd.DataFrame({
                "Source Estimate": self.source_eval_estimate[:mini_size].squeeze().numpy(),
                "Source Label": self.source_eval_label[:mini_size].squeeze().numpy(),
                "Target Estimate": self.target_eval_estimate[:mini_size].squeeze().numpy(),
                "Target Label": self.target_eval_label[:mini_size].squeeze().numpy()
            })
            output_dir = os.path.join(self.config.PROJECT_DIR, "results", f"S{self.subject_num}_adversarial_estimates_{suffix}.csv")
            df.to_csv(os.path.join(output_dir), index=False)