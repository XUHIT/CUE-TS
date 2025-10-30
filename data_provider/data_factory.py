
import torch
from torch.utils.data import DataLoader
from .custom_mts_dataset import CustomMTSDataset
import logging

logger = logging.getLogger(__name__)

def data_provider(args, flag):
    root_path = args.root_path
    data_path = args.data_path
    features = args.features
    target = args.target
    seq_len = args.seq_len
    pred_len = args.pred_len
    batch_size = args.batch_size

    logger.info(f"Creating dataset for {flag} with:")
    logger.info(f"  - Root path: {root_path}")
    logger.info(f"  - Data path: {data_path}")
    logger.info(f"  - Features: {features}")
    logger.info(f"  - Target: {target}")
    logger.info(f"  - Sequence length: {seq_len}")
    logger.info(f"  - Prediction length: {pred_len}")
    try:
        dataset = CustomMTSDataset(
            root_path=root_path,
            data_path=data_path,
            split=flag,
            context_length=seq_len,
            prediction_length=pred_len,
            stride=1,
            scale=True,
            features=features,
            target=target
        )
        
        logger.info(f"Successfully created {flag} dataset with {len(dataset)} samples")
        shuffle = (flag == 'train')  
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=(flag == 'train') 
        )
        
        logger.info(f"Created dataloader with batch_size={batch_size}, shuffle={shuffle}")
        return dataset, dataloader
        
    except Exception as e:
        logger.error(f"Error creating dataset for {flag}: {e}")
        raise e