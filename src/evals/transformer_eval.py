import os

import torch
import torchmetrics

from src.inference.greedy_search import greedy_decode


def run_validation(
    model,
    validation_ds,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_step,
    writer,
    num_examples=2,
):
    """
    Run validation on the provided dataset using the given model.

    Args:
        model: The trained model to use for validation.
        validation_ds: The dataset to validate against, expected to yield batches.
        tokenizer_tgt: Tokenizer for the target language.
        max_len: Maximum length for the decoded sequence.
        device: The device (CPU or GPU) to run the validation on.
        print_msg: Function to print messages, typically the console print function.
        global_step: The global step counter, used for logging.
        writer: An instance of SummaryWriter for logging metrics to TensorBoard.
        num_examples: The number of examples to print during validation.

    The function performs validation by decoding the input sequences using
    greedy decoding. It prints the source, target, and predicted translations
    for a specified number of examples. The function also computes and logs
    the character error rate (CER), word error rate (WER) and BLEU score using the
    torchmetrics library.
    """

    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_tgt,
                max_len,
                device,
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg("-" * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg("-" * console_width)
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar("validation cer", cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar("validation wer", wer, global_step)
        writer.flush()

        # TODO:Compute the BLEU metric
        # metric = torchmetrics.BLEUScore()
        # bleu = metric(predicted_list, expected_list)
        # writer.add_scalar("validation BLEU", bleu, global_step)
        # writer.flush()
