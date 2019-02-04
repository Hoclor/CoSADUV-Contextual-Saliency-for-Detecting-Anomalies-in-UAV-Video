import torch
import pickle

def main():
    from util.data_utils import get_SALICON_datasets
    from util.data_utils import get_direct_datasets

#     train_data, val_data, test_data, mean_image = get_SALICON_datasets('Dataset/Transformed') # 128x96
    dataset_root_dir = 'Dataset/Raw Dataset'
    mean_image_name = 'mean_image.npy'
    img_size = (96, 128) # height, width - original: 480, 640, reimplementation: 96, 128
    train_data, val_data, test_data = get_direct_datasets(dataset_root_dir, mean_image_name, img_size)

    def NSS_loss_alt(x, y):
        """
        Computes the Normalized Scanpath Saliency between x (output of a model)
        and y (label). X and Y are assumed to be torch tensors.
        """
        # Normalize x
        x = (x - x.mean())/x.std()
        # Create a binary mask to select values from x where the corresponding y value is > 0
        
        # Compute the element-wise multiplication of x and y
        scanpath = x * y
        
        # NSS = sum(scanpath)/sum(y)
        nss = scanpath.sum()/y.sum()
        
        # NSS loss = -NSS
        return -nss

    def NSS_loss(x, y):
        """
        Computes the Normalized Scanpath Saliency between x (output of a model)
        and y (label). X and Y are assumed to be torch tensors.
        """
        # Normalize x
        x = (x - x.mean())/x.std()
        # Create a binary mask to select values from x where the corresponding y value is > 0
        mask = y > 0 #Use threshold=0 always
        scanpath = torch.masked_select(x, mask)
        # return negative mean, as loss is minimized in training
        return -scanpath.mean()

    from models.DSCLRCN_PyTorch import DSCLRCN #DSCLRCN_PyTorch, DSCLRCN_PyTorch2 or DSCLRCN_PyTorch3
    from util.solver import Solver

    batchsize = 20 # Recommended: 20
    epoch_number = 10 # Recommended: 10 (epoch_number =~ batchsize/2)
    net_type = 'Seg' # 'Seg' or 'CNN' Recommended: Seg
    optim_str = 'SGD' # 'SGD' or 'Adam' Recommended: Adam
    optim_args = {'lr': 1e-2} # 1e-2 if SGD, 1e-4 if Adam
    loss_func = NSS_loss_alt # NSS_loss or torch.nn.KLDivLoss() Recommended: torch.nn.KLDivLoss()

    optim = torch.optim.SGD if optim_str == 'SGD' else torch.optim.Adam

    #num_train = 100
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=4)#,
                                            #sampler=OverfitSampler(num_train))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batchsize, shuffle=True, num_workers=4)

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

if __name__ == '__main__':
    # Note: You must put all your training code into one function rather than in the global scope
    #       (this is good practice anyway).
    #       Subsequently you must call the set_start_method and your main function from inside this
    #       if-statement. If you don't do that, each worker will attempt to run all of your training
    #       code and everything will go very wild and very wrong.
    torch.multiprocessing.set_start_method('forkserver') # spawn, forkserver, or fork
    print(torch.multiprocessing.get_start_method())
    main()
