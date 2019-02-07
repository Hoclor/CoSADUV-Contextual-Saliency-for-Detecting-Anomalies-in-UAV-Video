import torch
import pickle

def main():
    from util.data_utils import get_SALICON_datasets
    from util.data_utils import get_direct_datasets
    from tqdm import tqdm

#     train_data, val_data, test_data, mean_image = get_SALICON_datasets('Dataset/Transformed') # 128x96
    dataset_root_dir = 'Dataset/Raw Dataset'
    mean_image_name = 'mean_image.npy'
    img_size = (96, 128) # height, width - original: 480, 640, reimplementation: 96, 128
    train_data, val_data, test_data = get_direct_datasets(dataset_root_dir, mean_image_name, img_size)
    
    from models.DSCLRCN_PyTorch import DSCLRCN #DSCLRCN_PyTorch, DSCLRCN_PyTorch2 or DSCLRCN_PyTorch3
    from util.solver import Solver
    
    from util.loss_functions import NSS_loss

    batchsize = 20 # Recommended: 20
    epoch_number = 10 # Recommended: 10 (epoch_number =~ batchsize/2)
    net_type = 'Seg' # 'Seg' or 'CNN' Recommended: Seg
    optim_str = 'SGD' # 'SGD' or 'Adam' Recommended: Adam
    optim_args = {'lr': 1e-2} # 1e-2 if SGD, 1e-4 if Adam
    loss_func = NSS_loss # NSS_loss or torch.nn.KLDivLoss() Recommended: NSS_loss

    optim = torch.optim.SGD if optim_str == 'SGD' else torch.optim.Adam

    #num_train = 100
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True)#,
                                            #sampler=OverfitSampler(num_train))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True)

    # Attempt to train a model using the original image sizes
    model = DSCLRCN(input_dim=img_size, local_feats_net=net_type)
    # Set solver as torch.optim.SGD and lr as 1e-2, or torch.optim.Adam and lr 1e-4
    solver = Solver(optim=optim, optim_args=optim_args, loss_func=loss_func)
    solver.train(model, train_loader, val_loader, num_epochs=epoch_number, log_nth=50, filename_args={
        'batchsize' : batchsize,'epoch_number' : epoch_number,
        'net_type' : net_type, 'optim' : optim_str}
    )

    #Saving the model:
    model.save('trained_models/model_{}_{}_lr2_batch{}_epoch{}_model1'.format(net_type, optim_str, batchsize, epoch_number))
    with open('trained_models/solver_{}_{}_lr2_batch{}_epoch{}_model1.pkl'.format(net_type, optim_str, batchsize, epoch_number), 'wb') as outf:
        pickle.dump(solver, outf, pickle.HIGHEST_PROTOCOL)
    
    tqdm.write("Testing model and best checkpoint on SALICON validation set")
    
    # test on validation data as we don't have ground truths for the test data (this was also done in original DSCLRCN paper)
    test_losses = []
    test_loader = torch.utils.data.DataLoader(val_data, batch_size=20, shuffle=True, num_workers=4, pin_memory=True)
    for data in tqdm(test_loader):
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
        outputs = np.array([cv2.GaussianBlur(output, (int(4*sigma), int(4*sigma)), sigma) for output in outputs])

        # Compute the loss and append it to the list
        labels = labels.cpu().numpy()
        test_losses.append(NSS_loss(outputs, labels).item())
    
    # Delete the model to free up memory, load the best checkpoint of the model, and test this too
    del model
    filename = 'trained_models/best_model_{}_{}_lr2_batch{}_epoch{}.pth'.format(
                                net_type,
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
    model = DSCLRCN(input_dim=(96, 128), local_feats_net=net_type)
    model.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Test the checkpoint
    test_losses_checkpoint = []
    test_loader = torch.utils.data.DataLoader(val_data, batch_size=20, shuffle=True, num_workers=4, pin_memory=True)
    for data in tqdm(test_loader):
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
        outputs = np.array([cv2.GaussianBlur(output, (int(4*sigma), int(4*sigma)), sigma) for output in outputs])

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
    torch.multiprocessing.set_start_method('forkserver') # spawn, forkserver, or fork
    
    # Use CuDNN with benchmarking for performance improvement - from 1.05 batch20/s to 1.55 batch20/s on Quadro P4000
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    print("Using multiprocessing start method:", torch.multiprocessing.get_start_method())
    
    main()
