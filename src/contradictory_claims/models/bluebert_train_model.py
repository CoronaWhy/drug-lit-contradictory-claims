"""Functions to build, save, load and train the contradictory claims Transformer model based on bluebert."""

# -*- coding: utf-8 -*-

import datetime
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import AdamW, BertModel, BertTokenizer,\
    get_linear_schedule_with_warmup


class ContraDataset(Dataset):
    """Dataset loader."""

    def __init__(self, claims, labels, tokenizer, max_len=512, multi_class=True):
        """Initialize."""
        self.tokenizer = tokenizer
        self.claims = torch.tensor(self.regular_encode(claims, max_len=max_len, multi_class=multi_class))
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

    def regular_encode(self, texts, max_len=512, multi_class=True):
        """Tokenize a batch of sentence as an np.array."""
        if not multi_class:
            # If len > max_len, truncate text upto max_len-8 characters and append the 8-character auxillary input
            texts = [text[:max_len - 8] + text[-8:] if len(text) > max_len else text for text in texts]

        # texts = [text for text in texts if str(text) != 'nan']  # An annoying thing to catch... remove nans.
        enc_di = self.tokenizer.batch_encode_plus(texts,
                                                  return_token_type_ids=False,
                                                  padding='max_length',
                                                  max_length=max_len,
                                                  truncation=True)
        return np.array(enc_di['input_ids'])


class TorchContraNet(nn.Module):
    """Our transfer learning trained model."""

    def __init__(self, transformer, multi_class=True):
        """Define and initialize the layers of the model."""
        super(TorchContraNet, self).__init__()
        self.transformer = transformer
        if multi_class:
            self.linear = nn.Linear(768, 3)
            self.out = nn.Softmax(dim=1)
        else:
            self.linear = nn.Linear(768, 1)
            self.out = nn.Sigmoid()

    def forward(self, claim, mask, label=None):
        """Run the model on inputs."""
        hidden_states, enc_attn_mask = self.transformer(claim,
                                                        token_type_ids=None,
                                                        attention_mask=mask)
        unnormalized_labels = self.linear(hidden_states[:, 0, :])
        y = self.out(unnormalized_labels)
        return y

    def my_train(self):
        """Prepare transformer for training."""
        self.transformer.train()

    def eval(self):  # noqa: A003
        # Note: This runs fine, not sure if the A003 warning should be looked into
        """Freeze transformer weights."""
        self.transformer.eval()


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


def bluebert_create_model(bluebert_pretrained_path: str, multi_class: bool = True):
    """
    Create the Bluebert Transformer model.

    :param bluebert_pretrained_path: path to pretrained bluebert model
    :param multi_class: if True, final layer is multiclass so softmax is used. If False, final layer
        is sigmoid and binary crossentropy is evaluated.
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
    if multi_class:
        num_labels = 3
    else:
        num_labels = 2
    tokenizer = BertTokenizer.from_pretrained(bluebert_pretrained_path)
    transformer = BertModel.from_pretrained(bluebert_pretrained_path,
                                            num_labels=num_labels,
                                            output_attentions=False,
                                            output_hidden_states=False)

    # Create model
    model = TorchContraNet(transformer, multi_class=multi_class)
    model.my_train()
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
        torch_criterion = torch.nn.MSELoss(reduction='sum')
    elif criterion == 'crossentropy':
        torch_criterion = torch.nn.CrossEntropyLoss()
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

            print(f"Len claims {len(claim)}")
            print(f"Len mask {len(mask)}")

            if criterion == 'crossentropy':
                label = batch[2].to(device=device, dtype=torch.int64)
                label = torch.max(label, 1)[1]
            else:
                label = batch[2].to(device).float()

            model.zero_grad()

            y = model(claim, mask)
            loss = torch_criterion(y, label)
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
        print("  Training epoch took: ", format_time(time.time() - t0))  # noqa: T001

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
                                cord_train_x: np.ndarray,
                                cord_train_y: np.ndarray,
                                cord_test_x: np.ndarray,
                                cord_test_y: np.ndarray,
                                bluebert_pretrained_path: str,
                                use_multi_nli: bool = True,
                                use_med_nli: bool = True,
                                use_man_con: bool = True,
                                use_cord: bool = True,
                                epochs: int = 3,
                                batch_size: int = 32,
                                criterion: str = None,
                                multi_class: bool = True,):
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
    :param cord_train_x: CORD-19 training sentence pairs
    :param cord_train_y: CORD-19 training labels
    :param cord_test_x: CORD-19 test sentence pairs
    :param cord_test_y: CORD-19 test labels
    :param bluebert_pretrained_path: path to pretrained bluebert model, or huggingface model name
    :param multi_class: if True, final layer is multiclass so softmax is used. If False, final layer
        is sigmoid and binary crossentropy is evaluated.
    :param use_multi_nli: if True, use MultiNLI in fine-tuning
    :param use_med_nli: if True, use MedNLI in fine-tuning
    :param use_man_con: if True, use ManConCorpus in fine-tuning
    :param use_cord: if True, use CORD-19 in fine-tuning
    :param epochs: number of epochs for training
    :param batch_size: batch size for fine-tuning
    :param criterion: training loss criterion
    :return: fine-tuned Bluebert Transformer model
    :return device: CPU vs GPU definition for torch
    """
    # Create model
    model, tokenizer, device = bluebert_create_model(bluebert_pretrained_path, multi_class=multi_class)

    # Package data into a DataLoader
    if use_multi_nli:
        multinli_x_train_dataset = ContraDataset(list(multi_nli_train_x), multi_nli_train_y, tokenizer, max_len=512,
                                                 multi_class=multi_class)
        multinli_x_train_sampler = RandomSampler(multinli_x_train_dataset)
        multinli_x_train_dataloader = DataLoader(multinli_x_train_dataset,
                                                 sampler=multinli_x_train_sampler, batch_size=batch_size)

    if use_med_nli:
        mednli_x_train_dataset = ContraDataset(list(med_nli_train_x), med_nli_train_y, tokenizer, max_len=512,
                                               multi_class=multi_class)
        mednli_x_train_sampler = RandomSampler(mednli_x_train_dataset)
        mednli_x_train_dataloader = DataLoader(mednli_x_train_dataset, sampler=mednli_x_train_sampler,
                                               batch_size=batch_size)

    if use_man_con:
        mancon_x_train_dataset = ContraDataset(list(man_con_train_x), man_con_train_y, tokenizer, max_len=512,
                                               multi_class=multi_class)
        mancon_x_train_sampler = RandomSampler(mancon_x_train_dataset)
        mancon_x_train_dataloader = DataLoader(mancon_x_train_dataset, sampler=mancon_x_train_sampler,
                                               batch_size=batch_size)

    if use_cord:
        cord_x_train_dataset = ContraDataset(list(cord_train_x), cord_train_y, tokenizer, max_len=512,
                                             multi_class=multi_class)
        cord_x_train_sampler = RandomSampler(cord_x_train_dataset)
        cord_x_train_dataloader = DataLoader(cord_x_train_dataset, sampler=cord_x_train_sampler, batch_size=batch_size)

    losses_list = []

    # Fine tune model on MultiNLI
    if use_multi_nli:
        model, losses = bluebert_train_model(model, multinli_x_train_dataloader, device, epochs=epochs,
                                             criterion=criterion)
        losses_list.append(losses)
        print('Completed Bluebert fine tuning on MultiNLI')  # noqa: T001

    # Fine tune model on MedNLI
    if use_med_nli:
        model, losses = bluebert_train_model(model, mednli_x_train_dataloader, device, epochs=epochs,
                                             criterion=criterion)
        losses_list.append(losses)
        print('Completed Bluebert fine tuning on MedNLI')  # noqa: T001

    # Fine tune model on ManConCorpus
    if use_man_con:
        model, losses = bluebert_train_model(model, mancon_x_train_dataloader, device, epochs=epochs,
                                             criterion=criterion)
        losses_list.append(losses)
        print('Completed Bluebert fine tuning on ManConCorpus')  # noqa: T001

    # Fine tune model on CORD
    if use_cord:
        model, losses = bluebert_train_model(model, cord_x_train_dataloader, device, epochs=epochs,
                                             criterion=criterion)
        losses_list.append(losses)
        print('Completed Bluebert fine tuning on CORD')  # noqa: T001

    return model, losses, device


def bluebert_save_model(model, timed_dir_name: bool = True, bluebert_save_path: str = 'output/bluebert_transformer'):
    """
    Save fine-tuned Bluebert model.

    :param model: fine-tuned Bluebert model
    :param timed_dir_name: if True, save model to a directory where date is recorded
    :param bluebert_save_path: directory to save model
    """
    if timed_dir_name:
        now = datetime.datetime.now()
        bluebert_save_path = os.path.join(bluebert_save_path, f"{now.month}-{now.day}-{now.year}")

    if not os.path.exists(bluebert_save_path):
        os.makedirs(bluebert_save_path)

    bluebert_save_path = os.path.join(bluebert_save_path, "bluebert_model.pt")

    torch.save(model, bluebert_save_path)
    return


def bluebert_load_model(bluebert_model_path: str):
    """
    Load fine-tuned Bluebert model.

    :param bluebert_model_path: directory with saved model
    :return: fine-tuned Bluebert Transformer model
    :return device: CPU vs GPU definition for torch
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are ', torch.cuda.device_count(), ' GPU(s) available.')  # noqa: T001
        print('We will use the GPU:', torch.cuda.get_device_name(0))  # noqa: T001
    else:
        print('No GPU available, using the CPU instead.')  # noqa: T001
        device = torch.device("cpu")

    model = torch.load(bluebert_model_path, map_location=device)
    model.to(device)

    return model, device
