
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.auto import tqdm
import time
import shutil
import glob
from utils.plotting import plot_loss_curve, plot_step_loss
from utils.checkpoints import save_checkpoint, load_checkpoint
from qwen_vl_utils import process_vision_info

class QwenTrainer:
    def __init__(
        self,
        model,
        processor,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        optimizer = None,
        scheduler = None,
        config = None,
        output_dir = "output"
    ):
        self.model = model
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.device = model.device
        
        self.step_losses = []
        self.epoch_losses = []
        
        # Grad accum
        self.grad_accum = self.config.get("grad_accum", 1)

        # ------------------------------------------------------------------
        # FEATURE: Cast Trainable Params to FP32 (Stability)
        # ------------------------------------------------------------------
        self._cast_trainable_to_fp32()
        
        # ------------------------------------------------------------------
        # FEATURE: Detailed Parameter Accounting
        # ------------------------------------------------------------------
        self._log_model_stats()

    def _cast_trainable_to_fp32(self):
        """
        Cast all trainable parameters to float32.
        Required for stable mixed precision training (prevents gradient underflow).
        """
        print(f"[Init] Casting trainable parameters to float32...")
        count = 0
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)
                count += 1
        print(f"[Init] Cast {count} parameters to float32.")

    def _log_model_stats(self):
        """
        Log detailed parameter statistics (Frozen vs Trainable).
        """
        print(f"\n[param_stats] ========= Model Statistics =========")
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        for p in self.model.parameters():
            n_p = p.numel()
            total_params += n_p
            if p.requires_grad:
                trainable_params += n_p
            else:
                frozen_params += n_p
                
        print(f"  Total Params:     {total_params:,}")
        print(f"  Trainable Params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  Frozen Params:    {frozen_params:,}")
        print(f"==================================================\n")

    def train(self, epochs=1, resume_from=None):
        start_epoch = 0
        global_step = 0
        
        if resume_from:
            start_epoch, global_step = load_checkpoint(self.model, self.optimizer, self.scheduler, Path(resume_from))
            print(f"Resumed from epoch {start_epoch}, step {global_step}")
        
        self.model.train()
        
        for epoch in range(start_epoch, epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0
            steps_in_epoch = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            
            # Reset gradients at start of epoch
            self.optimizer.zero_grad()
            
            for step_idx, batch in enumerate(pbar):
                # Move batch to device
                inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward
                outputs = self.model(**inputs)
                loss = outputs.loss / self.grad_accum
                
                # Backward
                loss.backward()
                
                # Update weights every grad_accum steps
                if (step_idx + 1) % self.grad_accum == 0:
                    # Clip gradients if needed (optional, good practice)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # Logging (log full loss)
                    current_loss = loss.item() * self.grad_accum
                    self.step_losses.append(current_loss)
                    epoch_loss += current_loss
                    steps_in_epoch += 1 # Count effective steps
                    
                    pbar.set_postfix({"loss": f"{current_loss:.4f}"})
                    
                    # Save checkpoint every N steps (Using global_step)
                    if global_step > 0 and global_step % self.config.get("save_steps", 1000) == 0:
                        save_checkpoint(
                            self.model, self.processor, self.optimizer, self.scheduler,
                            epoch, global_step, current_loss, self.output_dir,
                            keep_last_n=self.config.get("keep_last_n", 3)
                        )
                        # Plot
                        if self.config.get("plot_every", 1) > 0:
                            plot_step_loss(self.step_losses, self.output_dir / "plots")
                            
                    # Inference Sampling (Placeholder for loop)
                    if global_step > 0 and global_step % self.config.get("inference_sampling_every", 999999) == 0:
                        self.run_inference_sampling(global_step)

            # Handle remaining gradients
            if (step_idx + 1) % self.grad_accum != 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1 # One last step
            
            avg_epoch_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0
            self.epoch_losses.append(avg_epoch_loss)
            print(f"Epoch finished. Avg Loss: {avg_epoch_loss:.4f}")
            
            # Run Validation
            val_loss = self.run_validation(epoch)
            
            # Save end of epoch
            save_checkpoint(
                self.model, self.processor, self.optimizer, self.scheduler,
                epoch + 1, global_step, avg_epoch_loss, self.output_dir, # Save as next epoch start
                keep_last_n=self.config.get("keep_last_n", 3)
            )
            plot_loss_curve(self.epoch_losses, [val_loss] if val_loss else [], [epoch+1] if val_loss else [], self.output_dir / "plots")
            
    def run_validation(self, epoch):
        if not self.val_loader:
            return None
        
        print("\n[Validation] Starting validation...")
        self.model.eval()
        total_val_loss = 0
        steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = self.model(**inputs)
                total_val_loss += outputs.loss.item()
                steps += 1
        
        avg_val_loss = total_val_loss / steps if steps > 0 else 0
        print(f"[Validation] Epoch {epoch+1} Loss: {avg_val_loss:.4f}")
        
        # Calculate Metrics (BLEU/CIDr) - TODO: Implement generation loop for metrics
        # For now, we only log loss to ensure pipeline runs
        
        self.model.train()
        return avg_val_loss
        
    def run_inference_sampling(self, step):
        """
        Run qualitative inference sampling and calculate metrics (BLEU-4, CIDr).
        Uses validation dataset subset for speed if large.
        """
        if not self.val_loader:
            return
            
        print(f"\n[Inference] Running sampling at step {step}...")
        self.model.eval()
        
        n_samples = self.config.get("inference_samples_n", 5)
        # 1. Select Random subset (or first N)
        # Note: val_loader is Shuffle=False usually, so we get consistent samples or random if shuffled.
        # We can just iterate.
        
        generated_captions = []
        ground_truths = []
        
        count = 0
        
        # Load Metrics
        from utils.metrics import compute_metrics
        
        with torch.no_grad():
            for batch in self.val_loader:
                if count >= n_samples: break
                
                # Unwrap Batch
                # Batch contains: "id", "messages", "image" (raw PIL), etc. if collator passes them
                # But QwenDataCollator processes 'inputs' -> input_ids, etc. 
                # We need raw prompts to generate.
                
                # Hack: QwenDataCollator in train.py typically returns tensors.
                # Use dataset directly for generation samples to avoid strict masking issues.
                pass
                
        # Since 'batch' in training loop is typically pre-processed tensors (labels already masked),
        # generating from them is hard because we need the Raw Text Prompt + Image to feed into .generate().
        # The standard approach is to pick index from dataset, re-process with generation template.
        
        if self.val_loader and hasattr(self.val_loader.dataset, "dataset"):
            # Subset -> Dataset
            ds_source = self.val_loader.dataset.dataset
        elif self.val_loader:
            ds_source = self.val_loader.dataset
            
        import random
        indices = list(range(len(ds_source)))
        random.shuffle(indices)
        indices = indices[:n_samples]
        
        print(f"Generating for {len(indices)} samples...")
        
        for idx in indices:
            item = ds_source[idx] # Returns {id, messages, image}
            
            # Extract inputs
            image = item["image"]
            messages = item["messages"] # User + Assistant
            
            # Construct Prompt (User only)
            user_msg = messages[0] # {role: user, content: [image, text]}
            gt_answer = messages[1]["content"][0]["text"]
            
            # Process for Inference
            text = self.processor.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
            
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            # max_new_tokens from config
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("inference_max_tokens", 128),
                do_sample=self.config.get("inference_do_sample", False),
                num_beams=self.config.get("inference_num_beams", 1),
                temperature=self.config.get("inference_temperature", 0.0)
            )
            
            # Trim input tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            print(f"\n[Sample {item['id']}]")
            print(f"Q: {user_msg['content'][1]['text']}")
            print(f"GT: {gt_answer}")
            print(f"Pred: {output_text}")
            
            generated_captions.append(output_text)
            ground_truths.append(gt_answer)
            
        # Calculate Metrics
        scores = compute_metrics(generated_captions, ground_truths)
        print(f"\n[Metrics] BLEU-4: {scores['bleu4']:.4f}")
                
        self.model.train()
