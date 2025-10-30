
import argparse
import logging
import torch
import sys
import os
import random
import numpy as np
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_factory import data_provider
from models.timemoe.modeling_moe_prompt import CovPromptTimeMoE, CovPromptConfig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler() 
        ]
    )
    return logging.getLogger(__name__)

def save_result(file_path, content):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(content + '\n')
        
def get_prediction_head_for_horizon(model, target_horizon, logger):
    pred_len = target_horizon
    
    if pred_len == 8:
        head_index = 1
    elif pred_len == 24 or pred_len == 32:
        head_index = 2
    elif pred_len == 64:
        head_index = 3
    else:
        # --- MODIFIED HERE ---
        raise ValueError(f"Unsupported prediction length {pred_len}. Only lengths 8, 24, 32, 64 are supported.")
    
    head_pred_len_map = {0: 1, 1: 8, 2: 32, 3: 64}
    
    try:
        head_pred_len = head_pred_len_map[head_index]
    except KeyError:
        raise ValueError(f"Unsupported head index: {head_index}")
    
    if pred_len > head_pred_len:
        # --- MODIFIED HERE ---
        raise ValueError(
            f"pred_len ({pred_len}) > head #{head_index}'s horizon ({head_pred_len}). "
            "Please select a head with a longer horizon or decrease the prediction_length."
        )
    
    selected_head = model.foundation_model.lm_heads[head_index]
    logger.info(f"Auto-selected Head #{head_index} (horizon {head_pred_len}) for prediction length {pred_len}")
    return selected_head, pred_len

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'constant':
        lr = args.learning_rate
    elif args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        lr = lr_adjust.get(epoch, args.learning_rate)
    else:
        lr = args.learning_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def evaluate_model_mse(model, dataloader, training_head, device, pred_len):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                inputs = batch['inputs'].to(device, dtype=torch.float16)
                labels = batch['labels'].to(device, dtype=torch.float16)
                variable_idx = batch['variable_idx'].to(device)
                
                target_var_idx = variable_idx[0].item()
                model_inputs = inputs[:, :, target_var_idx]
                all_variables_history = inputs
                
                hidden_states, _ = model(
                    inputs=model_inputs,
                    all_variables_history=all_variables_history, 
                    variable_idx=variable_idx
                )
                
                last_hidden_state = hidden_states[:, -1, :]
                preds = training_head(last_hidden_state)
                
                preds_truncated = preds[:, :pred_len]
                labels_truncated = labels[:, :pred_len]
                
                loss = torch.nn.functional.mse_loss(preds_truncated, labels_truncated)
                total_loss += loss.item() * inputs.shape[0]
                total_samples += inputs.shape[0]
            except Exception:
                continue
    
    return total_loss / total_samples if total_samples > 0 else float('inf')

def evaluate_model_test(model, dataloader, training_head, device, pred_len):
    model.eval()
    total_mse_loss = 0.0
    total_mae_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                inputs = batch['inputs'].to(device, dtype=torch.float16)
                labels = batch['labels'].to(device, dtype=torch.float16)
                variable_idx = batch['variable_idx'].to(device)
                
                target_var_idx = variable_idx[0].item()
                model_inputs = inputs[:, :, target_var_idx]
                all_variables_history = inputs
                
                hidden_states, _ = model(
                    inputs=model_inputs,
                    all_variables_history=all_variables_history,
                    variable_idx=variable_idx
                )
                
                last_hidden_state = hidden_states[:, -1, :]
                preds = training_head(last_hidden_state)
                
                preds_truncated = preds[:, :pred_len]
                labels_truncated = labels[:, :pred_len]
                
                mse_loss = torch.nn.functional.mse_loss(preds_truncated, labels_truncated)
                mae_loss = torch.nn.functional.l1_loss(preds_truncated, labels_truncated)
                
                batch_size = inputs.shape[0]
                total_mse_loss += mse_loss.item() * batch_size
                total_mae_loss += mae_loss.item() * batch_size
                total_samples += batch_size
            except Exception:
                continue
    
    if total_samples > 0:
        return total_mse_loss / total_samples, total_mae_loss / total_samples
    else:
        return float('inf'), float('inf')

def get_args():
    parser = argparse.ArgumentParser(description='CUE-TS Training for TimeMoE')
    parser.add_argument('--foundation_model_path', type=str, default='pretrained_model/TimeMoE-50M')
    parser.add_argument('--root_path', type=str, default='./datasets')
    parser.add_argument('--data_path', type=str, default='NP.csv')
    parser.add_argument('--target', type=str, default='-1')
    parser.add_argument('--seq_len', type=int, default=168)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-3)
    parser.add_argument('--lradj', type=str, default='constant', choices=['constant', 'type1'])
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--features', type=str, default='MS', choices=['MS'])
    parser.add_argument('--id_dim', type=int, default=32)
    parser.add_argument('--ts_dim', type=int, default=64)
    parser.add_argument('--attn_dim', type=int, default=256)
    parser.add_argument('--prompt_len', type=int, default=8)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--save_path', type=str, default='./checkpoints/')
    
    return parser.parse_args()

def main(args):
    if args.features != 'MS':
        raise ValueError(f"Only 'MS' feature type is supported, got '{args.features}'")
    set_seed(args.seed)
    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('./results', exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    dataset_name = args.data_path.split('.')[0]
    result_file = os.path.join('./results', f"{dataset_name}_S{args.seq_len}_P{args.pred_len}_PL{args.prompt_len}_AD{args.attn_dim}_LR{args.learning_rate}.txt")
    header = f"Data: {args.data_path}, Seq: {args.seq_len}, Pred: {args.pred_len}, PromptLen: {args.prompt_len}, AttnDim: {args.attn_dim}, LR: {args.learning_rate}, Batch: {args.batch_size}, Seed: {args.seed}"
    save_result(result_file, header)
    save_result(result_file, "-"*len(header)) 
    
    logger.info(f"Starting Training. Results will be saved to: {result_file}")
    
    train_dataset, train_dataloader = data_provider(args, flag='train')
    val_dataset, val_dataloader = data_provider(args, flag='val')
    test_dataset, test_dataloader = data_provider(args, flag='test')
    logger.info(f"Number of variables: {train_dataset.num_variables}")
    prompt_config = {
        'num_variables': train_dataset.num_variables,
        'id_dim': args.id_dim,
        'ts_dim': args.ts_dim,
        'attn_dim': args.attn_dim,
        'prompt_len': args.prompt_len,
        'model_dim': 384,
        'num_attn_heads': args.num_attn_heads
    }
    config = CovPromptConfig(
        foundation_model_path=args.foundation_model_path,
        prompt_config=prompt_config
    )

    model = CovPromptTimeMoE(config)
    model.to(device)
    training_head, actual_horizon = get_prediction_head_for_horizon(model, args.pred_len, logger)
    optimizer = Adam(
        model.prompt_generator.parameters(),
        lr=args.learning_rate,
    )     
    scaler = GradScaler() if args.use_amp and device.type == 'cuda' else None
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopped = False
    best_epoch = 0
    for epoch in range(args.num_epochs):
        adjust_learning_rate(optimizer, epoch + 1, args)
        if args.early_stopping and early_stopped:
            break
        model.train()
        epoch_train_loss = 0.0
        successful_batches = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            try:
                inputs = batch['inputs'].to(device, dtype=torch.float16)
                labels = batch['labels'].to(device, dtype=torch.float16)
                variable_idx = batch['variable_idx'].to(device)
                
                target_var_idx = variable_idx[0].item()
                model_inputs = inputs[:, :, target_var_idx]
                all_variables_history = inputs
                
                if scaler is not None:
                    with autocast():
                        hidden_states, _ = model(
                            inputs=model_inputs,
                            all_variables_history=all_variables_history,
                            variable_idx=variable_idx
                        )
                        last_hidden_state = hidden_states[:, -1, :]
                        preds = training_head(last_hidden_state)
                        preds_truncated = preds[:, :args.pred_len]
                        labels_truncated = labels[:, :args.pred_len]
                        loss = torch.nn.functional.mse_loss(preds_truncated, labels_truncated)
                    
                    scaler.scale(loss).backward()
                    if args.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.prompt_generator.parameters(), args.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    hidden_states, _ = model(
                        inputs=model_inputs,
                        all_variables_history=all_variables_history,
                        variable_idx=variable_idx
                    )
                    last_hidden_state = hidden_states[:, -1, :]
                    preds = training_head(last_hidden_state)
                    preds_truncated = preds[:, :args.pred_len]
                    labels_truncated = labels[:, :args.pred_len]
                    loss = torch.nn.functional.mse_loss(preds_truncated, labels_truncated)
                    
                    loss.backward()
                    if args.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.prompt_generator.parameters(), args.gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_train_loss += loss.item()
                successful_batches += 1
            except Exception:
                continue
        
        if successful_batches == 0:
            logger.warning(f"Epoch {epoch+1} had no successful batches. Skipping evaluation.")
            continue
        
        epoch_avg_train_loss = epoch_train_loss / successful_batches
        
        val_loss = evaluate_model_mse(model, val_dataloader, training_head, device, args.pred_len)
        test_mse, test_mae = evaluate_model_test(model, test_dataloader, training_head, device, args.pred_len)
        epoch_result = f"Epoch {epoch+1:02d} | Train MSE: {epoch_avg_train_loss:.6f} | Val MSE: {val_loss:.6f} | Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f}"
        logger.info(epoch_result)
        save_result(result_file, epoch_result)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            if args.early_stopping:
                save_path = os.path.join(args.save_path, f"best_model_{dataset_name}_S{args.seq_len}_P{args.pred_len}.pt")
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.prompt_generator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config,
                    'args': args,
                }
                torch.save(checkpoint_data, save_path)
                logger.info(f"Best model saved to: {save_path}")
        else:
            if args.early_stopping:
                patience_counter += 1
                if patience_counter >= args.patience:
                    early_stopped = True
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
    
    if not args.early_stopping:
        final_save_path = os.path.join(args.save_path, f"final_model_{dataset_name}_S{args.seq_len}_P{args.pred_len}.pt")
        final_checkpoint_data = {
            'epoch': args.num_epochs,
            'model_state_dict': model.prompt_generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config,
            'args': args,
        }
        torch.save(final_checkpoint_data, final_save_path)
        logger.info(f"Final model saved to: {final_save_path}")

    logger.info(f"Training completed. Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")

if __name__ == "__main__":
    args = get_args()
    main(args)