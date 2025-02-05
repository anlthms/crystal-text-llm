import torch
from transformers import Trainer
from datasets import Dataset, IterableDataset
from trl import GRPOConfig
import torch.nn.functional as F

class RLTrainer(Trainer):
    def __init__(self, model, reward_funcs, args, processing_class, train_dataset=None, eval_dataset=None):
        self.reward_funcs = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
        self.processing_class = processing_class
        self.num_generations = args.num_generations
        self.max_completion_length = args.max_completion_length
        self.temperature = args.temperature

        # define a simple data collator so Trainer doesn't complain
        def data_collator(features):
            return features

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=processing_class,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        input_ids_list = [entry["input_ids"].to(device) for entry in inputs]
        attention_mask_list = [entry["attention_mask"].to(device) for entry in inputs]
        prompts = [entry["prompt"] for entry in inputs]

        max_length = max(x.shape[0] for x in input_ids_list)
        # Pad input_ids & attention_mask
        input_ids = torch.stack([
            F.pad(x, (0, max_length - x.shape[0]), value=self.processing_class.pad_token_id)
            for x in input_ids_list
        ])
        attention_mask = torch.stack([
            F.pad(x, (0, max_length - x.shape[0]), value=0)
            for x in attention_mask_list
        ])

        batch_size = input_ids.size(0)
        # Generate multiple completions per prompt
        prompt_completion_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=self.temperature,
            num_return_sequences=self.num_generations
        )

        # Reshape [batch_size * num_generations, seq_len] -> grouped by prompt
        grouped = []
        for i in range(batch_size):
            start = i * self.num_generations
            end = start + self.num_generations
            grouped.append(prompt_completion_ids[start:end])

        # For each prompt, decode completions
        # Then compute group-based advantage
        all_losses = []
        for i in range(batch_size):
            group_ids = grouped[i]
            # slice out the newly generated tokens
            prompt_len = input_ids.shape[1]
            completion_ids_list = [g[prompt_len:] for g in group_ids]

            completions = [self.processing_class.decode(c, skip_special_tokens=True)
                           for c in completion_ids_list]

            # compute reward => a list of floats, one per completion
            # pass the same single prompt repeated, or store them in a list:
            single_prompt_list = [prompts[i]] * len(completions)
            group_rewards = self.reward_funcs[0](single_prompt_list, completions)

            # compute advantage
            # (You might do group-relative advantage or normalize across the entire batch.)
            # For example:
            rewards_tensor = torch.tensor(group_rewards, device=device, dtype=torch.float32)
            advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-4)

            # Next, compute the logprobs of those chosen tokens
            # We'll gather logprobs for each of those completions
            # Then multiply by advantages, sum, etc.

            # For each completion in the group:
            group_loss = []
            for j, cids in enumerate(completion_ids_list):
                # get logprobs from the model
                out = model(group_ids[j].unsqueeze(0))  # shape [1, seq_len, vocab_size]
                logprobs = out.logits.log_softmax(-1)

                # slice the new tokens
                new_tokens_logp = logprobs[:, prompt_len:, :]
                # gather for the actual tokens
                chosen_logp = new_tokens_logp.gather(2, cids.unsqueeze(0).unsqueeze(-1)).squeeze(-1)

                # negative sign for gradient ascent
                partial_loss = - chosen_logp.mean() * advantages[j]
                group_loss.append(partial_loss)

            # sum or average the group losses
            group_loss_val = torch.stack(group_loss).sum() / self.num_generations
            all_losses.append(group_loss_val)

        # final RL loss
        total_loss = torch.stack(all_losses).mean()
        return total_loss
