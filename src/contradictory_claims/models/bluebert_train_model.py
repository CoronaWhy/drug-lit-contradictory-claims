"""Functions to build, save, load and train the contradictory claims Transformer model based on bluebert."""

# -*- coding: utf-8 -*-

import datetime
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import AdamW, BertModel, BertTokenizer,\
    get_linear_schedule_with_warmup


class ContraDataset(Dataset):
    """Dataset loader."""

    def __init__(self, claims, labels, tokenizer, max_len=512):
        """Initialize."""
        self.tokenizer = tokenizer
        self.claims = torch.tensor(self.regular_encode(claims, max_len=max_len))
        self.att_mask = torch.zeros(self.claims.size())
        self.att_mask = torch.where(self.claims <= self.att_mask, self.att_mask, torch.ones(self.claims.size()))
        self.labels = labels

    def __getitem__(self, index):
        """Get data item from index."""
        # assert index < len(self.labels)
        return (self.claims[index], self.att_mask[index], torch.tensor(self.labels[index]))

    def __len__(self):
        """Get length of data."""
        return self.labels.shape[0]

    def regular_encode(self, texts, max_len=512):
        """Tokenize a batch of sentence as an np.array."""
        enc_di = self.tokenizer.batch_encode_plus(texts,
                                                  return_token_type_ids=False,
                                                  padding='max_length',
                                                  max_length=max_len,
                                                  truncation=True)
        return np.array(enc_di['input_ids'])


class TorchContraNet(nn.Module):
    """Our transfer learning trained model."""

    def __init__(self, transformer):
        """Define and initialize the layers of the model."""
        super(TorchContraNet, self).__init__()
        self.transformer = transformer
        self.linear = nn.Linear(768, 3)
        self.out = nn.Softmax(dim=0)

    def forward(self, claim, mask, label=None):
        """Run the model on inputs."""
        hidden_states, enc_attn_mask = self.transformer(claim,
                                                        token_type_ids=None,
                                                        attention_mask=mask)
        unnormalized_labels = self.linear(hidden_states[:, 0, :])
        y = self.out(unnormalized_labels)
        return y

    def train(self):
        """Prepare transformer for training."""
        self.transformer.train()

    # def eval(self):
        # """Freeze transformer weights."""
        # self.transformer.eval()


def format_time(elapsed):
    """
    Format time in seconds to hh:mm:ss.

    :param elapsed: time in seconds
    :return: Formatted time
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def bluebert_create_model(bluebert_pretrained_path: str):
    """
    Create the Bluebert Transformer model.

    :param bluebert_pretrained_path: path to pretrained bluebert model
    :return: pretrained Bluebert Transformer model
    :return device: CPU vs GPU definition for torch
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are ', torch.cuda.device_count(), ' GPU(s) available.')  # noqa: T001
        print('We will use the GPU:', torch.cuda.get_device_name(0))  # noqa: T001
    else:
        print('No GPU available, using the CPU instead.')  # noqa: T001
        device = torch.device("cpu")

    # Load pretrained bluebert transformer and tokenizer
    tokenizer = BertTokenizer.from_pretrained(bluebert_pretrained_path)
    transformer = BertModel.from_pretrained(bluebert_pretrained_path,
                                            num_labels=3,
                                            output_attentions=False,
                                            output_hidden_states=False)

    # Create model
    model = TorchContraNet(transformer)
    model.train()
    model.to(device)

    return model, tokenizer, device


def bluebert_train_model(model,
                         dataloader,
                         device,
                         criterion=None,
                         optimizer=None,
                         epochs: int = 3,
                         seed: int = 42):
    """
    Train the Bluebert Transformer model.

    :param model: Bluebert model definition
    :param dataloader: training data packaged into a dataloader
    :param device: CPU vs GPU definition for torch
    :param criterion: training loss criterion
    :param optimizer: training loss optimizer
    :param epochs: number of epochs for training
    :param seed: random seed value for initialization
    :return: fine-tuned Bluebert Transformer model
    """
    # Set training loss criterion and optimizer
    if criterion is None:
        criterion = torch.nn.MSELoss(reduction='sum')
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # Set the seed value all over the place to make this reproducible.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    loss_values = []

    # Initialize scheduler
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    # Training loop
    for epoch in range(epochs):
        print("")  # noqa: T001
        print('======== Epoch ', epoch + 1, ' / ', epochs, ' ========')  # noqa: T001
        print('Training Bluebert...')  # noqa: T001

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Step through dataloader output
        for step, batch in enumerate(dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch ', step, '  of  ', len(dataloader), '.    Elapsed: ', elapsed, '.')  # noqa: T001

            claim = batch[0].to(device)
            mask = batch[1].to(device)
            label = batch[2].to(device).float()

            model.zero_grad()

            y = model(claim, mask)
            loss = criterion(y, label)
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        print("")  # noqa: T001
        print("  Average training loss: ", avg_train_loss)  # noqa: T001
        print("  Training epcoh took: ", format_time(time.time() - t0))  # noqa: T001

    return model, loss_values


def bluebert_create_train_model(multi_nli_train_x: np.ndarray,
                                multi_nli_train_y: np.ndarray,
                                multi_nli_test_x: np.ndarray,
                                multi_nli_test_y: np.ndarray,
                                med_nli_train_x: np.ndarray,
                                med_nli_train_y: np.ndarray,
                                med_nli_test_x: np.ndarray,
                                med_nli_test_y: np.ndarray,
                                man_con_train_x: np.ndarray,
                                man_con_train_y: np.ndarray,
                                man_con_test_x: np.ndarray,
                                man_con_test_y: np.ndarray,
                                bluebert_pretrained_path: str,
                                use_multi_nli: bool = True,
                                use_med_nli: bool = True,
                                use_man_con: bool = True):
    """
    Create and train the Bluebert Transformer model.

    :param multi_nli_train_x: MultiNLI training sentence pairs
    :param multi_nli_train_y: MultiNLI training labels
    :param multi_nli_test_x: MultiNLI test sentence pairs
    :param multi_nli_test_y: MultiNLI test labels
    :param med_nli_train_x: MedNLI training sentence pairs
    :param med_nli_train_y: MedNLI training labels
    :param med_nli_test_x: MedNLI test sentence pairs
    :param med_nli_test_y: MedNLI test labels
    :param man_con_train_x: ManConCorpus training sentence pairs
    :param man_con_train_y: ManConCorpus training labels
    :param man_con_test_x: ManConCorpus test sentence pairs
    :param man_con_test_y: ManConCorpus test labels
    :param bluebert_pretrained_path: path to pretrained bluebert model
    :param use_multi_nli: if True, use MultiNLI in fine-tuning
    :param use_med_nli: if True, use MedNLI in fine-tuning
    :param use_man_con: if True, use ManConCorpus in fine-tuning
    :return: fine-tuned Bluebert Transformer model
    """
    # Create model
    model, tokenizer, device = bluebert_create_model(bluebert_pretrained_path)

    # Package data into a DataLoader
    multinli_x_train_dataset = ContraDataset(multi_nli_train_x.to_list(), multi_nli_train_y, tokenizer, max_len=512)
    multinli_x_train_sampler = RandomSampler(multinli_x_train_dataset)
    multinli_x_train_dataloader = DataLoader(multinli_x_train_dataset, sampler=multinli_x_train_sampler, batch_size=4)

    mednli_x_train_dataset = ContraDataset(med_nli_train_x.to_list(), med_nli_train_y, tokenizer, max_len=512)
    mednli_x_train_sampler = RandomSampler(mednli_x_train_dataset)
    mednli_x_train_dataloader = DataLoader(mednli_x_train_dataset, sampler=mednli_x_train_sampler, batch_size=4)

    mancon_x_train_dataset = ContraDataset(man_con_train_x.to_list(), man_con_train_y, tokenizer, max_len=512)
    mancon_x_train_sampler = RandomSampler(mancon_x_train_dataset)
    mancon_x_train_dataloader = DataLoader(mancon_x_train_dataset, sampler=mancon_x_train_sampler, batch_size=4)

    # Fine tune model on MultiNLI
    if use_multi_nli:
        model, losses = bluebert_train_model(model, multinli_x_train_dataloader, device)
    print('Completed Bluebert finw tuning on MultiNLI')  # noqa: T001

    # Fine tune model on MedNLI
    if use_med_nli:
        model, losses = bluebert_train_model(model, mednli_x_train_dataloader, device)
    print('Completed Bluebert finw tuning on MedNLI')  # noqa: T001

    # Fine tune model on ManConCorpus
    if use_man_con:
        model, losses = bluebert_train_model(model, mancon_x_train_dataloader, device)
    print('Completed Bluebert fine tuning on ManConCorpus')  # noqa: T001

    return model
