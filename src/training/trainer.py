
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
        self.val_step_losses = [] # Track val loss per step
        self.val_steps = []
        self.epoch_losses = []
        self.best_val_loss = float('inf')
        
        
        # Research-grade tracking
        self.lr_history = []        # Learning rate at each step
        self.grad_norms = []        # Gradient norms at each step
        self.epoch_stats = []       # Epoch-level statistics
        # Grad accum
        self.grad_accum = self.config.get("grad_accum", 1)
        
        # Debug mode
        self.debug = self.config.get("debug", False)
        if self.debug:
            print("[DEBUG MODE ENABLED] Verbose logging active")

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
            steps_per_epoch = len(self.train_loader) // self.grad_accum
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Steps in epoch: {steps_per_epoch}")
            print(f"  Global step: {global_step}")
            if self.debug:
                print(f"  [DEBUG] Train loader batches: {len(self.train_loader)}")
                print(f"  [DEBUG] Grad accum: {self.grad_accum}")
            print(f"{'='*60}")
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            
            epoch_loss = 0
            steps_in_epoch = 0
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
                    # Compute gradient norm BEFORE clipping (research)
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    self.grad_norms.append(total_norm)

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
                    # Track LR
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.lr_history.append(current_lr)
                    pbar.set_postfix({"loss": f"{current_loss:.4f}", "step": global_step, "lr": f"{current_lr:.2e}"})
                    
                    # Debug: Print every batch
                    if self.debug and global_step % 10 == 0:
                        lr = self.optimizer.param_groups[0]['lr']
                        print(f"\n[DEBUG] Step {global_step}: loss={current_loss:.4f}, lr={lr:.2e}")
                    
                    # Save checkpoint every N steps (Using global_step)
                    if global_step > 0 and global_step % self.config.get("save_steps", 1000) == 0:
                        save_checkpoint(
                            self.model, self.processor, self.optimizer, self.scheduler,
                            epoch, global_step, current_loss, self.output_dir,
                            keep_last_n=self.config.get("keep_last_n", 3)
                        )
                        # Plot
                        if self.config.get("plot_every", 1) > 0:
                            if self.debug:
                                print(f"[DEBUG] Saving checkpoint and plotting at step {global_step}")
                            plot_step_loss(
                                self.step_losses, 
                                self.output_dir / "plots", 
                                val_steps=self.val_steps, 
                                val_losses=self.val_step_losses
                            )
                    
                    # Periodic Validation (Step-level)
                    if self.config.get("val_every_steps") and global_step % self.config["val_every_steps"] == 0:
                        val_loss = self.run_validation(epoch)
                        if val_loss is not None:
                            self.val_step_losses.append(val_loss)
                            self.val_steps.append(global_step)
                            plot_step_loss(
                                self.step_losses, 
                                self.output_dir / "plots", 
                                val_steps=self.val_steps, 
                                val_losses=self.val_step_losses
                            )
                            
                            # Check for Best Checkpoint (Step Level)
                            if val_loss < self.best_val_loss:
                                print(f"New best validation loss: {val_loss:.4f} (was {self.best_val_loss:.4f})")
                                self.best_val_loss = val_loss
                                save_checkpoint(
                                    self.model, self.processor, self.optimizer, self.scheduler,
                                    epoch, global_step, val_loss, self.output_dir,
                                    keep_last_n=self.config.get("keep_last_n", 3),
                                    is_best=True
                                )
                                
                                
                        self.model.train()
                        torch.cuda.empty_cache()
                            
                    # Inference Sampling (Placeholder for loop)
                    if global_step > 0 and global_step % self.config.get("inference_sampling_every", 999999) == 0:
                        self.run_inference_sampling(epoch + 1, global_step)

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
            
            # Run Inference Sampling every N epochs
            inference_epoch_freq = self.config.get("inference_sampling_every_epochs", 5)
            if (epoch + 1) % inference_epoch_freq == 0:
                self.run_inference_sampling(epoch + 1, global_step)
            
            # Save end of epoch
            save_checkpoint(
                self.model, self.processor, self.optimizer, self.scheduler,
                epoch + 1, global_step, avg_epoch_loss, self.output_dir, # Save as next epoch start
                keep_last_n=self.config.get("keep_last_n", 3)
            )
            plot_loss_curve(self.epoch_losses, [val_loss] if val_loss else [], [epoch+1] if val_loss else [], self.output_dir / "plots")
            
            # Save Best Checkpoint
            if val_loss is not None and val_loss < self.best_val_loss:
                print(f"New best validation loss: {val_loss:.4f} (was {self.best_val_loss:.4f})")
                self.best_val_loss = val_loss
                save_checkpoint(
                    self.model, self.processor, self.optimizer, self.scheduler,
                    epoch + 1, global_step, val_loss, self.output_dir,
                    keep_last_n=self.config.get("keep_last_n", 3),
                    is_best=True
                )
            
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
        
    def run_inference_sampling(self, epoch, step):
        """
        Run qualitative inference sampling.
        """
        if not self.val_loader:
            return
            
        print(f"\n[Inference] Running sampling at epoch {epoch}, step {step}...")
        self.model.eval()
        torch.cuda.empty_cache() # Start with clean slate
        
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
        
        generated_captions = []
        ground_truths = []
        inference_results = []
        
        import json
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        for idx in indices:
            with torch.no_grad():
                item = ds_source[idx] # Returns {id, messages, image}
            
            # Extract inputs
            image = item["images"][0] # dataset returns list of images
            messages = item["messages"] # User + Assistant
            
            # Construct Prompt (Filter out Assistant answer)
            conversation = [msg for msg in messages if msg['role'] != 'assistant']
            gt_msg = next((msg for msg in reversed(messages) if msg['role'] == 'assistant'), None)
            gt_answer = gt_msg["content"][0]["text"] if gt_msg else "UNKNOWN"
            
            # Process for Inference
            text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("inference_max_tokens", 128),
                do_sample=self.config.get("inference_do_sample", False),
                num_beams=self.config.get("inference_num_beams", 1),
                temperature=self.config.get("inference_temperature", 0.0)
            )
            
            # Trim input tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Calculate Per-Sample Metrics
            chencherry = SmoothingFunction()
            # Basic whitespace tokenization for BLEU check
            ref_tokens = [gt_answer.split()] 
            hyp_tokens = output_text.split()
            bleu4 = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=chencherry.method1)
            
            # Log to file and console
            log_entry = f"\n[Sample {item['id']}]\nQ: {user_msg['content'][1]['text']}\nGT: {gt_answer}\nPred: {output_text}\nBLEU-4: {bleu4:.4f}\n"
            print(log_entry)
            
            # Save string log
            with open(self.output_dir / "inference_log.txt", "a") as f:
                f.write(log_entry)
            
            generated_captions.append(output_text)
            ground_truths.append(gt_answer)
            
            # Store Structured Result
            inference_results.append({
                "sample_token": item['id'],
                "prediction": output_text,
                "ground_truth": gt_answer,
                "metrics": {
                    "bleu4": bleu4,
                    "cider": 0.0 # Placeholder (Requires corpus-based stats)
                }
            })
            
        # Calculate Overall Metrics
        # Assuming compute_metrics is available via imports or utils
        try:
            from utils.metrics import compute_metrics
            scores = compute_metrics(generated_captions, ground_truths)
        except ImportError:
            scores = {"bleu4": 0.0, "cider": 0.0}
            
        metric_log = f"\n[Metrics] Overall BLEU-4: {scores.get('bleu4', 0):.4f}\n"
        print(metric_log)
        
        with open(self.output_dir / "inference_log.txt", "a") as f:
            f.write(metric_log)
            
        # Save Detailed JSON
        json_path = self.output_dir / f"inference_epoch_{epoch}.json"
        with open(json_path, "w") as f:
            json.dump({
                "epoch": epoch,
                "step": step,
                "overall_metrics": scores,
                "samples": inference_results
            }, f, indent=2)
            
        self.model.train()
        torch.cuda.empty_cache()
