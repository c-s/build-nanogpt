import os
import pathlib
import time
import numpy as np

import torch
from transformers import GPT2LMHeadModel

from trl import PPOTrainer
from trl import PPOConfig
from transformers import AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


from torch.utils.data import IterableDataset
from torch.nn import functional as F

torch.set_float32_matmul_precision('high')

expert_hf_model = GPT2LMHeadModel.from_pretrained(
    "gpt2",
    torch_dtype=torch.bfloat16,
    # is_traiable=True,
)
# student_hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
expert_model = expert_hf_model
# student_model = student_hf_model

device = "cuda"

expert_model.to(device)
# student_model.to(device)

use_compile = False  # np.int error...

if use_compile:
    expert_model = torch.compile(expert_model)
    # student_model = torch.compile(student_model)


stu_exp_kl_coef = 0.0  # 0.2
seq_length = 64
batch_size = 32
gradient_accumulation_steps = 1
disc_cross_entropy_factor = 0.02
reward_factor = 0.1

base_lr = 1.41e-5
disc_lr = 1.41e-6

gen_update_interval = 1

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class FineWebDataset(IterableDataset):
    def __init__(self, B, T, process_rank, num_processes, split):
        assert B == 1
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        # if master_process:
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

    def generate(self):
        while True:
            x, _ = self.next_batch()
            yield x[0]

    def __iter__(self):
        return iter(self.generate())


def build_config():
    ppo_config = PPOConfig(
        model_name="gpt2",
        learning_rate=base_lr,
        # learning_rate=1.41e-7,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mini_batch_size=batch_size // gradient_accumulation_steps,
        log_with="wandb",
        kl_penalty="full",
        # init_kl_coef=0.0,
        # adap_kl_ctrl=False,
        # accelerator_kwargs=dict(
        #     # device_placement=False,
        #     cpu=True,
        # )
    )

    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_config.model_name,
        torch_dtype=torch.bfloat16,
        is_trainable=True,
    )
    ppo_ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_config.model_name,
        torch_dtype=torch.bfloat16,
    )
    # ppo_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        # "eos_token_id": -1, #tokenizer.eos_token_id,
    }

    B, T = 1, seq_length
    dataset = FineWebDataset(B, T, 0, 1, "train")

    return dict(
        ppo_config=ppo_config,
        ppo_model=ppo_model,
        ppo_ref_model=ppo_ref_model,
        tokenizer=tokenizer,
        generation_kwargs=generation_kwargs,
        dataset=dataset,
    )


@torch.no_grad()
def get_gen_loss(
    expert_model,
    student_model,
    idx: torch.Tensor,
    initial_fixed_length: int,
):
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, T = idx.shape

        expert_pred = _get_qs(initial_fixed_length, idx, expert_model(idx)[0].detach())
        student_pred = _get_qs(initial_fixed_length, idx, student_model(idx)[0])

        D = expert_pred / (expert_pred + student_pred)

        loss = -torch.log(D) + torch.log(1 - D)

        return loss.mean()


def get_disc_loss(
    expert_model,
    student_model,
    expert_tokens: torch.Tensor,
    student_tokens: torch.Tensor,
    initial_fixed_length: int,
):
    _, T = expert_tokens.shape

    expert_path_expert_pred = _get_qs(
        initial_fixed_length, expert_tokens, expert_model(expert_tokens)[0]
    )
    expert_path_student_pred = _get_qs(
        initial_fixed_length, expert_tokens, student_model(expert_tokens)[0].detach()
    )
    student_path_expert_pred = _get_qs(
        initial_fixed_length, student_tokens, expert_model(student_tokens)[0]
    )
    student_path_student_pred = _get_qs(
        initial_fixed_length,
        student_tokens,
        student_model(student_tokens)[0].detach(),
    )

    expert_path_D = expert_path_expert_pred / (
        expert_path_expert_pred + expert_path_student_pred
    )
    student_path_D = student_path_expert_pred / (
        student_path_expert_pred + student_path_student_pred
    )

    kl_loss = stu_exp_kl_coef * (
        torch.log(student_path_student_pred) - torch.log(student_path_expert_pred)
    )

    cross_entropy_loss = -torch.log(1 - student_path_D) - torch.log(expert_path_D)
    cross_entropy_loss = cross_entropy_loss * disc_cross_entropy_factor

    return cross_entropy_loss.mean(), kl_loss.mean()


def _get_qs(
    initial_fixed_length: int, tokens: torch.Tensor, logits: torch.Tensor
) -> torch.Tensor:
    tokens = tokens[:, initial_fixed_length:]  # (B, T')
    # (B, T', vocab_size)
    logits = logits[:, initial_fixed_length - 1 : -1, :]
    predictions = F.softmax(logits, dim=-1)
    return torch.gather(predictions, 2, tokens[:, :, None])[:, :, 0]


# initial_fixed_length_scenario = LengthSampler(8, T - 2)


def initial_fixed_length_scenario(i):
    # # return 32
    max_length = seq_length - 1
    min_length = max(16, min(max_length, max_length - int(max_length * (i / 500))))
    length = torch.randint(min_length, max_length + 1, (1, )).item()
    return length

# def initial_fixed_length_scenario(i):
#     # # return 32
#     m = T - 1
#     return max(16, min(m, m - int(m * (i / 500))))

def train(resource):
    ppo_config = resource["ppo_config"]
    ppo_model = resource["ppo_model"]
    ppo_ref_model = resource["ppo_ref_model"]
    tokenizer = resource["tokenizer"]
    generation_kwargs = resource["generation_kwargs"]
    dataset = resource["dataset"]

    ppo_optimizer = torch.optim.AdamW(
        ppo_model.parameters(), lr=base_lr, betas=(0.9, 0.95), eps=1e-8
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=ppo_model,
        ref_model=ppo_ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        # optimizer=ppo_optimizer,
    )

    disc_optimizer = torch.optim.AdamW(
        expert_model.parameters(), lr=disc_lr, betas=(0.9, 0.95), eps=1e-8
    )

    total_index = 0
    while True:
        for i, batch in enumerate(ppo_trainer.dataloader):
            t0 = time.time()
            # initial_fixed_length = initial_fixed_length_scenario()
            initial_fixed_length = initial_fixed_length_scenario(total_index)
            # gen_loss, disc_loss = irl.get_loss(tokens, 3, use_expert_rollout=True)
            # print(f"step: {i} gen_loss: {gen_loss}, disc_loss: {disc_loss}")
            max_new_tokens = batch.size(1) - initial_fixed_length
            query_tensor = batch[:, :initial_fixed_length]
            response_tensor = torch.full_like(batch, tokenizer.eos_token_id)

            ppo_trainer.model.eval()
            response_list = ppo_trainer.generate(
                list(query_tensor),
                **dict(generation_kwargs, max_new_tokens=max_new_tokens),
            )
            # ppo_trainer.model.train()

            for res_ind, res in enumerate(response_list):
                response_tensor[res_ind, : len(res)] = res

            expert_model.train()
            expert_model.zero_grad()
            student_model = ppo_trainer.model
            disc_loss_accum = 0.0
            kl_loss_accum = 0.0
            micro_batch_size = batch_size // gradient_accumulation_steps
            for micro_step in range(gradient_accumulation_steps):
                start_batch_index = micro_batch_size * micro_step
                end_batch_index = start_batch_index + micro_batch_size
                micro_batch = batch[start_batch_index:end_batch_index]
                micro_response_tensor = response_tensor[start_batch_index:end_batch_index]
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    micro_disc_loss, micro_kl_loss = get_disc_loss(
                        expert_model,
                        student_model,
                        micro_batch,
                        micro_response_tensor,
                        initial_fixed_length,
                    )
                    micro_disc_loss = micro_disc_loss / gradient_accumulation_steps
                    micro_kl_loss = micro_kl_loss / gradient_accumulation_steps
                    disc_loss_accum += micro_disc_loss.detach()
                    kl_loss_accum += micro_kl_loss.detach()
                    micro_sum_loss = micro_disc_loss + micro_kl_loss
                    micro_sum_loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(expert_model.parameters(), 1.0)
            disc_optimizer.step()
            # disc_loss = 0.0

            if total_index % gen_update_interval == 0:
                def get_rewards(response_tensor) -> torch.Tensor:
                    expert_logits = expert_model(response_tensor)[0].detach()
                    student_logits = student_model(response_tensor)[0]

                    expert_prob = _get_qs(
                        initial_fixed_length, response_tensor, expert_logits
                    )
                    student_prob = _get_qs(
                        initial_fixed_length, response_tensor, student_logits
                    )

                    D = expert_prob / (expert_prob + student_prob)

                    # kl_loss = stu_exp_kl_coef * (torch.log(student_prob) - torch.log(expert_prob))
                    kl_loss = 0
                    reward = torch.log(D) - torch.log(1 - D) - kl_loss
                    reward = reward * reward_factor

                    return reward.mean(dim=-1)

                rewards = []
                for micro_step in range(gradient_accumulation_steps):
                    start_batch_index = micro_batch_size * micro_step
                    end_batch_index = start_batch_index + micro_batch_size
                    micro_response_tensor = response_tensor[start_batch_index:end_batch_index]
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        rewards.extend(get_rewards(micro_response_tensor))
                # rewards = [get_reward(student_response) for student_response in response_tensor]

                stats = ppo_trainer.step(
                    list(query_tensor),
                    list(response_tensor[:, initial_fixed_length:]),
                    rewards,
                )
                log_batch = dict(
                    query=list(list(x) for x in query_tensor),
                    response=list(list(x) for x in batch[:, initial_fixed_length:]),
                )
                ppo_trainer.log_stats(stats, log_batch, rewards)

                gen_loss = get_gen_loss(
                    expert_model, student_model, response_tensor, initial_fixed_length
                )

                t1 = time.time()
                dt = t1 - t0
                tokens_processed = batch_size * seq_length
                tokens_per_sec = tokens_processed / dt
                avg_reward = sum(r.item() for r in rewards) / len(rewards)
                print(
                    f"step: {i}, initial_fixed_length: {initial_fixed_length}, gen_loss: {gen_loss:.3f}, disc_loss: {disc_loss_accum:.3f}, reward: {avg_reward:.4f}, norm: {norm:.4f}, dt: {dt:.2f}s | tok/sec: {tokens_per_sec:.2f}"
                )

            if total_index % 1000 == 0:
                checkpoint = {
                    'expert_model': expert_model.state_dict(),
                    'student_model': ppo_model.state_dict(),
                    'ppo_config': ppo_config,
                    'step': total_index,
                    'gen_loss': gen_loss.item(),
                    'disc_loss': disc_loss_accum.item(),
                }
                checkpoint_path = pathlib.Path(f"log/model_{total_index:05d}.pt")
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
            total_index += 1


def main():
    resource = build_config()
    train(resource)


if __name__ == "__main__":
    main()
