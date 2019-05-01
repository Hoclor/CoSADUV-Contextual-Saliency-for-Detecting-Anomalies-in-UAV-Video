def main():
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision
    from torch.autograd import Variable
    import torch.nn as nn
    import pickle
    from random import randint, randrange
    import sys
    from tqdm import tqdm
    import cv2

    print("CUDA available: {}".format(torch.cuda.is_available()))

    location = "ncc"

    # Import model architectures
    from models.DSCLRCN_OldContext import DSCLRCN
    from models.CoSADUV import CoSADUV
    from models.CoSADUV_NoTemporal import CoSADUV_NoTemporal

    # Prepare settings and get the datasets
    from util.data_utils import get_SALICON_datasets, get_video_datasets

    ### Data options ###
    dataset_root_dir = "Dataset/UAV123"  # Dataset/[SALICON, UAV123]
    mean_image_name = (
        "mean_image.npy"
    )  # Must be located at dataset_root_dir/mean_image_name
    img_size = (
        480,
        640,
    )  # height, width - original: 480, 640, reimplementation: 96, 128
    duration = (
        300
    )  # Length of sequences loaded from each video, if a video dataset is used

    from util import loss_functions

    from util.solver import Solver

    ### Testing options ###

    # Minibatchsize: Determines how many images are processed at a time on the GPU
    minibatchsize = 1  # Recommended: 4 for 480x640 for >12GB mem, 2 for <12GB mem.

    # Loss functions:
    # From loss_functions (use loss_functions.LOSS_FUNCTION_NAME)
    # NSS_loss
    # CE_MAE_loss
    # PCC_loss
    # KLDiv_loss
    loss_func = loss_functions.NSS_alt  # Recommended: NSS_loss
    test_loss_func = loss_functions.CE_MAE_loss

    ########## PREPARE DATASETS ##########

    ### Prepare datasets and loaders ###

    if "SALICON" in dataset_root_dir:
        train_data, val_data, test_data, mean_image = get_SALICON_datasets(
            dataset_root_dir, mean_image_name, img_size
        )
        train_loader = [
            torch.utils.data.DataLoader(
                train_data,
                batch_size=minibatchsize,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
            )
        ]
        val_loader = [
            torch.utils.data.DataLoader(
                val_data,
                batch_size=minibatchsize,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
            )
        ]
        # Load test loader using val_data as SALICON does not give GT for its test set
        test_loader = [
            torch.utils.data.DataLoader(
                val_data,
                batch_size=minibatchsize,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
            )
        ]
    elif "UAV123" in dataset_root_dir:
        train_loader, val_loader, test_loader, mean_image = get_video_datasets(
            dataset_root_dir,
            mean_image_name,
            duration=duration,
            img_size=img_size,
            shuffle=False,
            loader_settings={
                "batch_size": minibatchsize,
                "num_workers": 8,
                "pin_memory": False,
            },
        )

    ########## LOADING MODELS ##########

    # Loading a model from the saved state that produced
    # the lowest validation loss during training:

    # Requires the model classes be loaded

    # Assumes the model uses models.CoSADUV_NoTemporal architecture.
    # If not, this method will fail
    def load_model_from_checkpoint(model_name):
        filename = "trained_models/" + model_name + ".pth"
        if torch.cuda.is_available():
            checkpoint = torch.load(filename)
        else:
            # Load GPU model on CPU
            checkpoint = torch.load(filename, map_location="cpu")
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_accuracy"]

        if "DSCLRCN" in model_name:
            model = DSCLRCN(input_dim=img_size, local_feats_net="Seg")
        elif "CoSADUV_NoTemporal" in model_name:
            model = CoSADUV_NoTemporal(input_dim=img_size, local_feats_net="Seg")
        elif "CoSADUV" in model_name:
            model = CoSADUV(input_dim=img_size, local_feats_net="Seg")
        else:
            tqdm.write("Error: no model name found in filename: {}".format(model_name))
            return
        # Ignore extra parameters ('.num_batches_tracked'
        # that are added on NCC due to different pytorch version)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        tqdm.write(
            "=> loaded model checkpoint '{}' (trained for {} epochs)\n   with architecture {}".format(
                model_name, checkpoint["epoch"], type(model).__name__
            )
        )

        if torch.cuda.is_available():
            model = model.cuda()
            tqdm.write("   loaded to cuda")
        model.eval()
        return model

    def load_model(model_name):
        model = torch.load("trained_models/" + model_name, map_location="cpu")
        print("=> loaded model '{}'".format(model_name))
        if torch.cuda.is_available():
            model = model.cuda()
            print("   loaded to cuda")
        model.eval()
        return model

    ########## LOAD THE MODELS ##########
    models = []
    model_names = []
    # Loading some pretrained models to test them on the images:

    # DSCLRCN models
    ## Trained on SALICON
    ### NSS_loss
    # model_names.append("DSCLRCN/SALICON NSS -1.62NSS val best and last/best_model_DSCLRCN_NSS_loss_batch20_epoch5")
    ## Trained on UAV123
    ### NSS_alt loss func
    # model_names.append("DSCLRCN/UAV123 NSS_alt 1.38last 3.15best testing/best_model_DSCLRCN_NSS_alt_batch20_epoch5")

    # CoSADUV_NoTemporal models
    ## Trained on UAV123
    ### DoM loss func
    model_names.append(
        "CoSADUV_NoTemporal/DoM SGD 0.01lr - 3.16 NSS_alt/best_model_CoSADUV_NoTemporal_DoM_batch20_epoch6"
    )
    ### NSS_alt loss func
    # model_names.append("CoSADUV_NoTemporal/NSS_alt Adam lr 1e-4 - 1.36/best_model_CoSADUV_NoTemporal_NSS_alt_batch20_epoch5")
    ### CE_MAE loss func
    # model_names.append("CoSADUV_NoTemporal/best_model_CoSADUV_NoTemporal_CE_MAE_loss_batch20_epoch10")

    # CoSADUV models (CoSADUV2)
    ## Trained on UAV123
    ### NSS_alt loss func
    #### 1 Frame backpropagation
    #### Kernel size 1
    # model_names.append("CoSADUV/NSS_alt Adam 0.001lr 1frame backprop size1 kernel -2train -0.7val 1epoch/best_model_CoSADUV_NSS_alt_batch20_epoch5")
    #### Kernel size 3
    model_names.append(
        "CoSADUV/NSS_alt Adam 0.01lr 1frame backprop size3 kernel/best_model_CoSADUV_NSS_alt_batch20_epoch5"
    )
    #### 2 Frame backpropagation
    #### Kernel size 3
    model_names.append(
        "CoSADUV/NSS_alt Adam 0.01lr 2frame backprop size3 kernel - 6.56 NSS_alt val/best_model_CoSADUV_NSS_alt_batch20_epoch5"
    )
    ### DoM loss func
    # Only very poor results achieved
    ### CE_MAE loss func
    # Only very poor results achieved

    max_name_len = max([len(name) for name in model_names])
    # Load the models specified above
    iterable = model_names

    # for i, name in enumerate(iterable):
    #    if "best_model" in name:
    #        models.append(load_model_from_checkpoint(name))
    #    else:
    #        models.append(load_model(name))

    print()
    print("Loaded all specified models")

    ########## TEST THE MODEL ##########

    # Define a function for testing a model
    # Output is resized to the size of the data_source
    def test_model(model, data_loader, loss_fns=[loss_functions.MAE_loss]):
        loss_sums = []
        loss_counts = []
        for i, loss_fn in enumerate(loss_fns):
            if loss_fn != loss_functions.NSS_alt:
                loss_sums.append(0)
                loss_counts.append(0)
            else:
                loss_sums.append([0, 0])
                loss_counts.append([0, 0])

        loop1 = data_loader
        if location != "ncc":
            loop1 = tqdm(loop1)

        for video_loader in loop1:
            # Reset temporal state if model is temporal
            if model.temporal:
                model.clear_temporal_state()

            loop2 = video_loader
            if location != "ncc":
                loop2 = tqdm(loop2)

            for data in loop2:
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # Produce the output
                outputs = model(inputs).squeeze(1)
                # if model is temporal detach its state
                if model.temporal:
                    model.detach_temporal_state()
                # Move the output to the CPU so we can process it using numpy
                outputs = outputs.cpu().data.numpy()

                # If outputs contains a single image, insert
                # a singleton batchsize dimension at index 0
                if len(outputs.shape) == 2:
                    outputs = np.expand_dims(outputs, 0)

                # Resize the images to input size
                outputs = np.array(
                    [
                        cv2.resize(output, (labels.shape[2], labels.shape[1]))
                        for output in outputs
                    ]
                )

                outputs = torch.from_numpy(outputs)

                if torch.cuda.is_available():
                    outputs = outputs.cuda()
                    labels = labels.cuda()
                # Apply each loss function, add results to corresponding entry in loss_sums and loss_counts
                for i, loss_fn in enumerate(loss_fns):
                    # If loss fn is NSS_alt, manually add std_dev() if the target is all-0
                    if loss_fn == loss_functions.NSS_alt:
                        for i in range(len(labels)):
                            if labels[i].sum() == 0:
                                loss_sums[i][1] += outputs[i].std().item()
                                loss_counts[i][1] += 1
                            else:
                                loss_sums[i][0] += loss_fn(outputs[i], labels[i]).item()
                                loss_counts[i][0] += 1
                    else:
                        loss_sums[i] += loss_fn(outputs, labels).item()
                        loss_counts[i] += 1

        return loss_sums, loss_counts

    # Obtaining loss values on the test set for different models:
    loop3 = model_names
    if location != "ncc":
        loop3 = tqdm(loop3)

    for i, model_name in enumerate(loop3):
        tqdm.write("model name: {}".format(model_name))
        if "best_model" in model_name:
            model = load_model_from_checkpoint(model_name)
        else:
            model = load_model(model_name)

        loss_fns = [
            loss_functions.NSS_alt,
            loss_functions.CE_MAE_loss,
            loss_functions.CE_loss,
            loss_functions.MAE_loss,
            loss_functions.DoM,
        ]

        test_losses, test_counts = test_model(model, val_loader, loss_fns=loss_fns)

        # Print out the result

        tqdm.write("[{}] Model: ".format(i, model_names[i]))

        for i, func in enumerate(loss_fns):
            if func == loss_functions.NSS_alt:
                tqdm.write(
                    ("{:25} : {:6f}").format(
                        "NSS_alt (+ve imgs)", test_losses[i][0] / test_counts[i][0]
                    )
                )
                tqdm.write(
                    ("{:25} : {:6f}").format(
                        "NSS_alt (-ve imgs)", test_losses[i][1] / test_counts[i][1]
                    )
                )
            else:
                tqdm.write(
                    ("{:25} : {:6f}").format(
                        func.__name__, test_losses[i] / test_counts[i]
                    )
                )
        del model


if __name__ == "__main__":
    import torch

    torch.multiprocessing.set_start_method("forkserver")  # spawn, forkserver, or fork

    # Use CuDNN with benchmarking for performance improvement:
    # from 1.05 batch20/s to 1.55 batch20/s on Quadro P4000
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    main()
