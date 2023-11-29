import torch

from tqdm import tqdm

import data_loader.data_loaders as module_data
import global_config

# fix random seeds for reproducibility
from train import _set_seed
from utils.optimize_thresholds import optimize_ts

def fine_tune_thresholds(config, cv_data_dir=None, valid_idx=None, k_fold=None):

    import model.final_model as module_arch

    _set_seed(global_config.SEED)

    logger = config.get_logger('fine_tune_threshold_fold_' + str(k_fold))

    data_loader = getattr(module_data, config['data_loader']['type'])(
            cv_data_dir,
            batch_size=64,
            shuffle=False,
            validation_split=0.0,
            num_workers=4,
            cross_valid=True,
            test_idx=valid_idx,
            cv_train_mode=False,
            fold_id=k_fold
    )

    class_labels = data_loader.dataset.class_labels

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # Load the model from the checkpoint
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # Prepare the model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Inference: Compute the raw network scores on the validation data
    with torch.no_grad():

        # Store the intermediate targets. Always store the output scores
        outputs_list = []
        targets_list = []

        for batch_idx, (padded_records, _, first_labels, labels_one_hot, record_names) in \
                enumerate(tqdm(data_loader)):
            data, target = padded_records.to(device), labels_one_hot.to(device)

            data = data.permute(0, 2, 1)  # switch seq_len and feature_size (12 = #leads)
            output = model(data)

            if type(output) is tuple:
                output, attention_weights = output

            # Detach tensors needed for further tracing and metrics calculation to remove them from the graph
            detached_output = output.detach().cpu()
            detached_target = target.detach().cpu()

            outputs_list.append(detached_output)
            targets_list.append(detached_target)

    # Get detached tensors from the list for fine-tuning the thresholds
    # For this, create a tensor from the dynamically filled list
    det_outputs = torch.cat(outputs_list).detach().cpu()
    det_targets = torch.cat(targets_list).detach().cpu()

    # ------------ Fine tune thresholds ------------------------------------
    thresholds = optimize_ts(target=det_targets, logits=det_outputs, labels=class_labels)

    return thresholds
