def main():
    import pickle

    import numpy as np
    from torch.autograd import Variable
    from tqdm import tqdm

    import cv2
    from models.CoSADUV_NoTemporal import CoSADUV_NoTemporal
    from util import loss_functions
    from util.data_utils import get_SALICON_datasets, get_video_datasets
    from util.solver import Solver

    location = "ncc"  # ncc or '', where the code is to be run (affects output)
    if location == "ncc":
        print_func = print
    else:
        print_func = tqdm.write

    ### Data options ###

    dataset_root_dir = "Dataset/UAV123"  # Dataset/[SALICON, UAV123]
    # Name of mean_image file: Must be located at dataset_root_dir/mean_image_name
    mean_image_name = "mean_image.npy"
    # Height, width of images
    img_size = (480, 640)  # Original: 480, 640, reimplementation: 96, 128
    # Duration: Length of sequences loaded from each video, if a video dataset is used
    duration = 300

    ### Training options ###

    # Batchsize: Determines how many images are processed before backpropagation is done
    batchsize = 20  # Recommended: 20.
    # Minibatchsize: Determines how many images are processed at a time on the GPU
    minibatchsize = 2  # Recommended: 4 for 480x640 for >12GB mem, 2 for <12GB mem.
    epoch_number = 5  # Recommended: 10 (epoch_number =~ batchsize/2)
    optim_str = "SGD"  # 'SGD' or 'Adam' Recommended: Adam
    optim_args = {"lr": 1e-2}  # 1e-2 if SGD, 1e-2 if Adam
    # Loss functions:
    # From loss_functions (use loss_functions.LOSS_FUNCTION_NAME)
    # NSS_loss
    # CE_MAE_loss
    # PCC_loss
    # KLDiv_loss
    loss_func = loss_functions.DoM  # Recommended: NSS_loss
    test_loss_func = loss_functions.NSS_alt

    ### Prepare optimiser ###
    if batchsize % minibatchsize:
        print(
            "Error, batchsize % minibatchsize must equal 0 ({} % {} != 0).".format(
                batchsize, minibatchsize
            )
        )
        exit()
    num_minibatches = batchsize / minibatchsize

    # Scale the lr down as smaller minibatches are used
    optim_args["lr"] /= num_minibatches

    optim = torch.optim.SGD if optim_str == "SGD" else torch.optim.Adam

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
    else:
        # Assume dataset is in VideoDataset structure
        train_loader, val_loader, test_loader, mean_image = get_video_datasets(
            dataset_root_dir,
            mean_image_name,
            duration=duration,
            img_size=img_size,
            shuffle=True,
            loader_settings={
                "batch_size": minibatchsize,
                "num_workers": 8,
                "pin_memory": False,
            },
        )

    ### Training ###

    model = CoSADUV_NoTemporal(input_dim=img_size, local_feats_net="Seg")

    print("Starting train on model with settings:")
    print("### Dataset settings ###")
    print("Dataset: {}".format(dataset_root_dir.split("/")[-1]))
    print("Image size: ({}h, {}w)".format(img_size[0], img_size[1]))
    print("Sequence duration: {}".format(duration))
    print("")
    print("### Training settings ###")
    print("Batch size: {}".format(batchsize))
    print("Minibatch size: {}".format(minibatchsize))
    print("Epochs: {}".format(epoch_number))
    print("")
    print("### Optimiser settings ###")
    print("Optimiser: {}".format(optim_str))
    print("Effective lr: {}".format(str(optim_args["lr"] * num_minibatches)))
    print("Actual lr: {}".format(str(optim_args["lr"])))
    print("Loss function: {}".format(loss_func.__name__))
    print("Model: {}".format(type(model).__name__))
    print("\n")

    # Create a solver with the options given above and appropriate location
    solver = Solver(
        optim=optim, optim_args=optim_args, loss_func=loss_func, location=location
    )
    # Start training
    solver.train(
        model,
        train_loader,
        val_loader,
        num_epochs=epoch_number,
        num_minibatches=num_minibatches,
        log_nth=50,
        filename_args={"batchsize": batchsize, "epoch_number": epoch_number},
    )

    # Saving the model:
    model_name = "{}_{}_batch{}_epoch{}".format(
        type(model).__name__, loss_func.__name__, batchsize, epoch_number
    )
    model.save("trained_models/model_" + model_name)
    with open("trained_models/solver_" + model_name + ".pkl", "wb") as outf:
        pickle.dump(solver, outf, pickle.HIGHEST_PROTOCOL)

    ### Testing ###

    print_func("Testing model")
    print_func("(on val set if using SALICON, otherwise on test set)\n")

    test_loss, test_count = test_model(
        model, test_loader, test_loss_func, location=location
    )

    # Delete the model to free up memory
    del model
    filename = "trained_models/best_model_" + model_name + ".pth"

    # Load the checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location="cpu")
    start_epoch = checkpoint["epoch"]
    # Create the model
    model = CoSADUV_NoTemporal(input_dim=img_size, local_feats_net="Seg")
    model.load_state_dict(checkpoint["state_dict"])
    if torch.cuda.is_available():
        model = model.cuda()

    # Test the checkpoint
    print_func(
        "Testing best checkpoint, after {} epochs of training".format(start_epoch)
    )
    test_loss_checkpoint, test_count_checkpoint = test_model(
        model, test_loader, test_loss_func, location=location
    )

    # Print out the result
    print()
    print("{} score on test set:".format(test_loss_func.__name__))
    print("(Higher is better)")
    print("Last model     : {:6f}".format(-1 * test_loss / test_count))
    print(
        "Best Checkpoint: {:6f}".format(
            -1 * test_loss_checkpoint / test_count_checkpoint
        )
    )


def test_model(model, test_set, loss_fn, location="ncc"):
    from tqdm import tqdm
    import numpy as np
    import cv2
    import torch

    # Set the model to evaluation mode
    model.eval()

    loss = 0
    count = 0
    test_loop = test_set
    if location != "ncc":
        test_loop = tqdm(test_loop, desc="Test (best checkpoint)")
    for video_loader in test_loop:
        if location != "ncc":
            video_loader = tqdm(video_loader, desc="Video")

        # If the model is temporal, reset its temporal state
        # at the start of each video
        if model.temporal:
            model.clear_temporal_state()

        for data in video_loader:
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

            loss += loss_fn(outputs, labels).item()
            count += 1
    return loss, count


if __name__ == "__main__":
    import torch

    torch.multiprocessing.set_start_method("forkserver")  # spawn, forkserver, or fork

    # Use CuDNN with benchmarking for performance improvement:
    # from 1.05 batch20/s to 1.55 batch20/s on Quadro P4000
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    main()
