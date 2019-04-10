def main():
    import pickle

    import numpy as np
    from torch.autograd import Variable
    from tqdm import tqdm

    import cv2
    from util.data_utils import (get_dataloader, get_nvvl_UAV123_datasets,
                                get_SALICON_datasets)

    location = 'ncc' # ncc or '', where the code is to be run (affects output)
    if location == 'ncc':
        print_func = print
    else:
        print_func = tqdm.write

#     train_data, val_data, test_data, mean_image = get_SALICON_datasets('Dataset/Transformed') # 128x96
    dataset_root_dir = 'Dataset/SALICON'
    mean_image_name = 'mean_image.npy'
    img_size = (480, 640) # height, width - original: 480, 640, reimplementation: 96, 128

    if 'SALICON' in dataset_root_dir:
        dataset_type = 'SALICON'
        train_data, val_data, test_data = get_SALICON_datasets(dataset_root_dir, mean_image_name, img_size)
    elif 'UAV123' in dataset_root_dir:
        dataset_type = 'UAV123'
        train_data, train_targets, val_data, val_targets, test_data, test_targets = get_nvvl_UAV123_datasets(dataset_root_dir, shuffle=True, sequence_length=150, img_size=img_size)

    
    from models.DSCLRCN_PyTorch import DSCLRCN #DSCLRCN_PyTorch, DSCLRCN_PyTorch2 or DSCLRCN_PyTorch3
    from util.solver import Solver
    
    from util.loss_functions import NSS_loss

    batchsize = 20 # Recommended: 20. Determines how many images are processed before backpropagation is done
    minibatchsize = 4 # Recommended: 4 for 480x640 for 12GB mem, 2 for 8GB mem. Determines how many images are processed in parallel on the GPU at once
    epoch_number = 20 # Recommended: 10 (epoch_number =~ batchsize/2)
    optim_str = 'SGD' # 'SGD' or 'Adam' Recommended: Adam
    optim_args = {'lr': 1e-2} # 1e-2 if SGD, 1e-4 if Adam
    loss_func = NSS_loss # NSS_loss or torch.nn.KLDivLoss() Recommended: NSS_loss

    if batchsize % minibatchsize:
        print("Error, batchsize % minibatchsize must equal 0 ({} % {} != 0).".format(batchsize, minibatchsize))
        exit()
    num_minibatches = batchsize/minibatchsize

    optim = torch.optim.SGD if optim_str == 'SGD' else torch.optim.Adam

    train_loader = get_dataloader(dataset_type, train_data, batch_size=minibatchsize, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = get_dataloader(dataset_type, val_data, batch_size=minibatchsize, shuffle=True, num_workers=4, pin_memory=True)

    # Attempt to train a model using the original image sizes
    model = DSCLRCN(input_dim=img_size, local_feats_net='Seg')
    # Set solver as torch.optim.SGD and lr as 1e-2, or torch.optim.Adam and lr 1e-4
    solver = Solver(optim=optim, optim_args=optim_args, loss_func=loss_func, location='ncc')
    solver.train(model, train_loader, val_loader, num_epochs=epoch_number, num_minibatches=num_minibatches, log_nth=50, 
        filename_args={'batchsize' : batchsize, 'epoch_number' : epoch_number, 'optim' : optim_str}
    )

    #Saving the model:
    model.save('trained_models/model_{}_lr2_batch{}_epoch{}'.format(optim_str, batchsize, epoch_number))
    with open('trained_models/solver_{}_lr2_batch{}_epoch{}.pkl'.format(optim_str, batchsize, epoch_number), 'wb') as outf:
        pickle.dump(solver, outf, pickle.HIGHEST_PROTOCOL)
    
    print_func("Testing model and best checkpoint on SALICON validation set")
    
    test_losses = []
    if dataset_type == 'SALICON':
        # SALICON does not provide GT for test set, so test on val set instead
        test_loader = get_dataloader(val_data, batch_size=minibatchsize, shuffle=True, num_workers=8, pin_memory=True)
    else:
        test_loader = get_dataloader(test_data, batch_size=minibatchsize, shuffle=True, num_workers=8, pin_memory=True)

    looper=test_loader
    if location != 'ncc':
        looper=tqdm(looper)

    for data in looper:
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

        # Produce the output
        outputs = model(inputs).squeeze()
        # Move the output to the CPU so we can process it using numpy
        outputs = outputs.cpu().data.numpy()

        # Resize the images to input size
        outputs = np.array([cv2.resize(output, (labels.shape[2], labels.shape[1])) for output in outputs])

        # Apply a Gaussian filter to blur the saliency maps
        sigma = 0.035*min(labels.shape[1], labels.shape[2])
        kernel_size = int(4*sigma)
        # make sure the kernel size is odd
        kernel_size += 1 if kernel_size % 2 == 0 else 0
        
        outputs = np.array([cv2.GaussianBlur(output, (kernel_size, kernel_size), sigma) for output in outputs])

        # Compute the loss and append it to the list
        labels = labels.cpu().numpy()
        test_losses.append(NSS_loss(outputs, labels).item())
    
    # Delete the model to free up memory, load the best checkpoint of the model, and test this too
    del model
    filename = 'trained_models/best_model_{}_lr2_batch{}_epoch{}.pth'.format(
                                optim_str,
                                batchsize,
                                epoch_number)
    
    # Load the checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    
    # Create the model
    model = DSCLRCN(input_dim=img_size, local_feats_net='Seg')
    model.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()
        
    print("START TEST.")
    
    # Test the checkpoint
    test_losses_checkpoint = []
    if dataset_type == 'SALICON':
        # SALICON does not provide GT for test set, so test on val set instead
        test_loader = get_dataloader(val_data, batch_size=minibatchsize, shuffle=True, num_workers=8, pin_memory=True)
    else:
        test_loader = get_dataloader(test_data, batch_size=minibatchsize, shuffle=True, num_workers=8, pin_memory=True)
    
    looper = test_loader
    if location != 'ncc':
        looper = tqdm(looper)
    
    for data in looper:
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

        # Produce the output
        outputs = model(inputs).squeeze()
        # Move the output to the CPU so we can process it using numpy
        outputs = outputs.cpu().data.numpy()

        # Resize the images to input size
        outputs = np.array([cv2.resize(output, (labels.shape[2], labels.shape[1])) for output in outputs])

        # Apply a Gaussian filter to blur the saliency maps
        sigma = 0.035*min(labels.shape[1], labels.shape[2])
        kernel_size = int(4*sigma)
        # make sure the kernel size is odd
        kernel_size += 1 if kernel_size % 2 == 0 else 0
        
        outputs = np.array([cv2.GaussianBlur(output, (kernel_size, kernel_size), sigma) for output in outputs])
        
        # Compute the loss and append it to the list
        labels = labels.cpu().numpy()
        test_losses_checkpoint.append(NSS_loss(outputs, labels).item())

    # Print out the result
    print()
    print('NSS on Validation set:')
    print('Last model     : {:6f}'.format(-1*np.mean(test_losses)))
    print('Best Checkpoint: {:6f}'.format(-1*np.mean(test_losses_checkpoint)))


if __name__ == '__main__':
    # Note: You must put all your training code into one function rather than in the global scope
    #       (this is good practice anyway).
    #       Subsequently you must call the set_start_method and your main function from inside this
    #       if-statement. If you don't do that, each worker will attempt to run all of your training
    #       code and everything will go very wild and very wrong.
    import torch
    torch.multiprocessing.set_start_method('forkserver') # spawn, forkserver, or fork
    
    # Use CuDNN with benchmarking for performance improvement - from 1.05 batch20/s to 1.55 batch20/s on Quadro P4000
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    print("Using multiprocessing start method:", torch.multiprocessing.get_start_method())
    
    main()
