from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config.transformer_config import (
    get_config,
    get_weights_file_path,
    latest_weights_file_path,
)
from src.dataset.transformer_ds import get_ds
from src.evals.transformer_eval import run_validation
from src.models.transformer.transformer import Transformer, TransformerEmbedding
from src.train.utils import get_device


def build_model(config, tokenizer_src, tokenizer_tgt, device):
    src_emb = TransformerEmbedding(
        d_model=config["d_model"],
        max_len=config["max_len"],
        vocab_size=tokenizer_src.get_vocab_size(),
        drop_prob=config["drop_prob"],
        device=device,
    )

    tgt_emb = TransformerEmbedding(
        d_model=config["d_model"],
        drop_prob=config["drop_prob"],
        max_len=config["max_len"],
        vocab_size=tokenizer_tgt.get_vocab_size(),
        device=device,
    )

    model = Transformer(
        src_emb,
        tgt_emb,
        tokenizer_tgt.get_vocab_size(),
        config["d_model"],
        config["n_head"],
        config["d_hidden"],
        config["n_layers"],
        device,
        config["drop_prob"],
    ).to(device)

    # Initialize the parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.0
    ).to(device)

    return model, optimizer, loss_fn


def preload_model(model, optimizer, config):
    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    model_filename = (
        latest_weights_file_path(config)
        if preload == "latest"
        else get_weights_file_path(config, preload)
        if preload
        else None
    )
    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")

    return model, optimizer, initial_epoch, global_step


def train_model(config):
    # Define the device
    device = get_device()
    print("Using device:", device.type)

    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(
            f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB"
        )

    # Make sure the weights folder exists
    Path(f"src/weights/{config['datasource']}_{config['model_folder']}").mkdir(
        parents=True, exist_ok=True
    )

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model, optimizer, loss_fn = build_model(
        config, tokenizer_src, tokenizer_tgt, device
    )

    model, optimizer, initial_epoch, global_step = preload_model(
        model, optimizer, config
    )

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (B, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)  # (B, 1, seq_len, seq_len)
            label = batch["label"].to(device)  # (B, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            proj_output = model(
                encoder_input, decoder_input, encoder_mask, decoder_mask
            )  # (B, seq_len, vocab_size)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )

            # Log the loss
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(
            model,
            val_dataloader,
            tokenizer_tgt,
            config["max_len"],
            device,
            lambda msg: batch_iterator.write(msg),
            global_step,
            writer,
        )

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )
