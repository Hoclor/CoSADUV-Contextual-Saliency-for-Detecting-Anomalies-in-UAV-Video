def main():
    # NoTemporal DoM:
    # 2.2 frames/s on GeForce RTX 2080 Ti, 10989 MB memory, using 3200 MB GPU memory and 3700 MB RAM, with
    import pickle

    import numpy as np
    from torch.autograd import Variable
    from tqdm import tqdm

    import cv2
    from models.CoSADUV import CoSADUV
    from models.CoSADUV_NoTemporal import CoSADUV_NoTemporal
    from models.DSCLRCN import DSCLRCN
    from util import loss_functions
    from util.data_utils import get_SALICON_datasets, get_video_datasets
    from util.solver import Solver

    location = ""  # ncc or '', where the code is to be run (affects output)
    if location == "ncc":
        print_func = print
    else:
        print_func = tqdm.write

    ### General options ###

    # Height, width of images
    img_size = (480, 640)  # Original: 480, 640, reimplementation: 96, 128
    # Duration: Length of sequences loaded from each video, if a video dataset is used
    duration = 10000

    ### Model options ###
    model_text = """Models available for demonstration (at index i):
        DSCLRCN (all with NSS_alt loss function):
            [0] trained on SALICON
            [1] trained on UAV123
        CoSADUV_NoTemporal (all trained on UAV123):
            [2] DoM loss function (best CoSADUV_NT)
            [3] NSS_alt loss function
            [4] CE_MAE loss function
        CoSADUV (all trained on UAV123, with NSS_alt loss function):
            [5] 1x1 convLSTM kernel, 1 frame backprop
            [6] 3x3 convLSTM kernel, 1 frame backprop (best CoSADUV)
            [7] 3x3 convLSTM kernel, 2 frame backprop
        CoSADUV_NoTemporal + Transfer Learning (UAV123 + EyeTrackUAV):
            [8] DoM loss function
            [9] NSS_alt loss function\n"""
    print(model_text)
    model_index = int(input("Model index: (0-9): "))

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

    # List of all models available 
    models = []

    # DSCLRCN models
    ## Trained on SALICON
    ### NSS_loss
    models.append("DSCLRCN/SALICON NSS -1.62NSS val best and last/best_model_DSCLRCN_NSS_loss_batch20_epoch5")
    ## Trained on UAV123
    ### NSS_alt loss func
    models.append("DSCLRCN/UAV123 NSS_alt 1.38last 3.15best testing/best_model_DSCLRCN_NSS_alt_batch20_epoch5")

    # CoSADUV_NoTemporal models
    ## Trained on UAV123
    ### DoM loss func
    models.append("CoSADUV_NoTemporal/DoM SGD 0.01lr - 3.16 NSS_alt/best_model_CoSADUV_NoTemporal_DoM_batch20_epoch6")
    ### NSS_alt loss func
    models.append("CoSADUV_NoTemporal/NSS_alt Adam lr 1e-4 - 1.36/best_model_CoSADUV_NoTemporal_NSS_alt_batch20_epoch5")
    ### CE_MAE loss func
    models.append("CoSADUV_NoTemporal/best_model_CoSADUV_NoTemporal_CE_MAE_loss_batch20_epoch10")

    # CoSADUV models (CoSADUV2)
    ## Trained on UAV123
    ### NSS_alt loss func
    #### 1 Frame backpropagation
    #### Kernel size 1
    models.append("CoSADUV/NSS_alt Adam 0.001lr 1frame backprop size1 kernel -2train -0.7val 1epoch/best_model_CoSADUV_NSS_alt_batch20_epoch5")
    #### Kernel size 3
    models.append("CoSADUV/NSS_alt Adam 0.01lr 1frame backprop size3 kernel/best_model_CoSADUV_NSS_alt_batch20_epoch5")
    #### 2 Frame backpropagation
    #### Kernel size 3
    models.append(
        "CoSADUV/NSS_alt Adam 0.01lr 2frame backprop size3 kernel - 6.56 NSS_alt val/best_model_CoSADUV_NSS_alt_batch20_epoch5"
    )

    # Transfer learned models
    models.append("best_model_CoSADUV_NoTemporal_DoM_batch20_epoch5")
    models.append("best_model_CoSADUV_NoTemporal_NSS_alt_batch20_epoch5")

    model_name = models[model_index]

    if "best_model" in model_name:
        model = load_model_from_checkpoint(model_name)
    else:
        model = load_model(model_name)

    ### Data options ###

    print()
    dataset_name = input("Dataset (UAV123/EyeTrackUAV): ")

    if dataset_name == "UAV123":
        sequences_string = """bike [1-3]
boat [1-9]
car [1-18]
group [1-3]
person [1-23]
truck [1-4]
wakeboard [1-10]"""
        print(sequences_string)
        sequence_name = input("Sequence name: ")

    if dataset_name == "EyeTrackUAV":
        sequences_string = """bike3
boat [6, 8]
car [2, 4, 6, 7, 8, 10, 13]
group2
person [3, 13, 14, 18, 20]
truck1
wakeboard10"""
        print(sequences_string)
        sequence_name = input("Sequence name: ")

    if dataset_name not in ["SALICON", "UAV123", "EyeTrackUAV"]:
        print_func("Error: unrecognized dataset '{}'".format(dataset_name))
        exit()

    dataset_root_dir = "Dataset/" + dataset_name  # Dataset/[SALICON, UAV123]
    # Name of mean_image file: Must be located at dataset_root_dir/mean_image_name
    mean_image_name = "mean_image.npy"

    ### Testing options ###

    # Minibatchsize: Determines how many images are processed at a time on the GPU
    minibatchsize = 1  # Recommended: 4 for 480x640 for >12GB mem, 2 for <12GB mem.
    # Loss functions:
    # From loss_functions (use loss_functions.LOSS_FUNCTION_NAME)
    # NSS_loss
    # CE_MAE_loss
    # PCC_loss
    # KLDiv_loss
    loss_funcs = [loss_functions.NSS_alt, loss_functions.MAE_loss, loss_functions.DoM]  # Recommended: NSS_loss

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
                test_data,
                batch_size=minibatchsize,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
            )
        ]
    else:
        # Assume dataset is in VideoDataset structure
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

    ### Testing ###
    def test_model(model, test_set, sequence_name="", loss_fns=[], location="ncc"):
        # Get the index of the sequence
        seq_index = test_set.get_videos().index(sequence_name)
        if seq_index < 0:
            print("Error: sequence '{}' not found".format(sequence_name))
            quit()
        # Set the model to evaluation mode
        model.eval()
        losses = [0 for _ in loss_fns]
        counts = [0 for _ in loss_fns]
        loop1 = test_set
        # Only use tqdm here if we loop through all videos
        if location != "ncc" and sequence_name == "":
            loop1 = tqdm(loop1)
        for j, video_loader in enumerate(loop1):
            # Skip videos until the one named sequence_name is found
            if sequence_name != "" and j < seq_index:
                continue
            if j > seq_index:
                break
            loop2 = video_loader
            if location != "ncc":
                loop2 = tqdm(loop2, desc=sequence_name)

            # If the model is temporal, reset its temporal state
            # at the start of each video
            if model.temporal:
                model.clear_temporal_state()

            for i, data in enumerate(loop2):
                # Reset the model for each image if dataset is SALICON
                if dataset_name == "SALICON" and model.temporal:
                    model.clear_temporal_state()
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                # Produce the output
                outputs = model(inputs).squeeze(1)
                if inputs.shape != labels.shape:
                    # Move the output to the CPU so we can process it using numpy
                    outputs = outputs.cpu().data.numpy()
                    # Resize the images to labels size
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

                # Scale output to [0, 1] if model is temporal
                if model.temporal:
                    outputs -= outputs.min()
                    outputs /= outputs.max()

                vid_loss = [loss_fn(outputs, labels).item() for loss_fn in loss_fns]
                print_func("Frame [{}]".format(i))
                for j in range(len(vid_loss)):
                    losses[j] += vid_loss[j]
                    counts[j] += 1
                    print_func("    {:8}: {:4f}".format(loss_fns[j].__name__, vid_loss[j]))
                # Show the input, GT, prediction with cv2
                _input = inputs.squeeze().cpu().data.numpy().transpose(1, 2, 0) + mean_image
                _input = cv2.cvtColor(_input, cv2.COLOR_RGB2BGR)
                label = cv2.cvtColor(labels.squeeze().cpu().data.numpy(), cv2.COLOR_GRAY2BGR)
                output = cv2.cvtColor(outputs.squeeze().cpu().data.numpy(), cv2.COLOR_GRAY2BGR)
                # Add a white vertical line between each image
                vert = np.ones((480, 3, 3))*255
                out = np.hstack((_input, vert, label, vert, output))
                cv2.imshow("Output", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cv2.destroyAllWindows()
        return losses, counts

    if dataset_name == "SALICON":
        print_func("Running val set")
        test_model_loader = val_loader
    else:
        # Get the correct loader for the requested sequence
        if sequence_name in train_loader.get_videos():
            print("Running sequence {} in train set".format(sequence_name))
            test_model_loader = train_loader
        elif sequence_name in val_loader.get_videos():
            print("Running sequence {} in val set".format(sequence_name))
            test_model_loader = val_loader
        elif sequence_name in test_loader.get_videos():
            print("Running sequence {} in test set".format(sequence_name))
            test_model_loader = test_loader

    test_losses, test_counts = test_model(
        model, test_model_loader, sequence_name, loss_funcs, location=location
    )

    # Print out the result
    print("Mean scores")
    for i, loss_fn in enumerate(loss_funcs):
        if test_counts[i] > 0:
            print("{}: {}".format(loss_fn.__name__, test_losses[i] / test_counts[i]))


if __name__ == '__main__':
    import torch

    torch.multiprocessing.set_start_method("spawn")  # spawn, forkserver, or fork

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    main()
