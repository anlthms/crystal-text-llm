import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from datasets import Dataset, IterableDataset
from trl import GRPOConfig
import torch.nn.functional as F

class RLTrainer(Trainer):
    def __init__(self, model, reward_funcs, args, processing_class, train_dataset=None, eval_dataset=None):
        self.reward_funcs = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
        self.processing_class = processing_class
        self.num_generations = args.num_generations
        self.beta = args.beta

        def data_collator(features):  # No data collation is needed in GRPO
            return features

        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, processing_class=processing_class)

    def compute_loss_unused(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # compute GRPO loss
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [self.processing_class(example, return_tensors="pt", padding=True, truncation=True) for example in prompts]
        prompt_inputs = super()._prepare_inputs(prompts_text)

        prompt_completion_ids = model.generate(**prompt_inputs, max_new_tokens=self.args.max_completion_length)
        completion_ids = prompt_completion_ids[:, prompt_inputs["input_ids"].size(1):]
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        rewards = torch.tensor([self.reward_funcs[0](prompts, completions)], dtype=torch.float32, device=device)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        per_token_logps = model(prompt_completion_ids).logits.log_softmax(dim=-1)
        loss = -(per_token_logps * advantages.unsqueeze(1)).sum() / completions.size(1)

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("RLTrainer does not support returning outputs")

        device = self.accelerator.device

        # Extract inputs
        input_ids_list = [entry["input_ids"].to(device) for entry in inputs]
        attention_mask_list = [entry["attention_mask"].to(device) for entry in inputs]
        prompts = [entry["prompt"] for entry in inputs]

        # Find max sequence length
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

        # Generate completions
        prompt_completion_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.args.max_completion_length,
            do_sample=True,
            temperature=self.args.temperature
        )

        # Extract completions
        prompt_length = input_ids.shape[1]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Compute rewards
        rewards = torch.tensor(self.reward_funcs[0](prompts, completions), dtype=torch.float32, device=device)

        # Normalize rewards
        mean_rewards = rewards.mean()
        std_rewards = rewards.std() + 1e-4
        advantages = (rewards - mean_rewards) / std_rewards  # Shape: (batch_size,)

        # Compute per-token log probabilities
        per_token_logps = model(prompt_completion_ids).logits.log_softmax(dim=-1)

        # Expand advantages to match per-token shape
        token_length = per_token_logps.shape[1]  # e.g., 367
        advantages = advantages.unsqueeze(1).expand(-1, token_length)  # Shape: (4, 367)

        # Compute GRPO loss
        loss = -(per_token_logps[:, :, :completion_ids.shape[-1]] * advantages.unsqueeze(2)).sum() / completion_ids.shape[1]

        return loss
