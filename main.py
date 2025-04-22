from getData import getData
from setup import setup_main
from test import testing
from train import training

def main():
    setup_main()
    assets = getData()

    training(
        model=assets.model,
        criterion=assets.criterion,
        scheduler=assets.scheduler,
        num_epochs=assets.num_epochs,
        train_loader=assets.train_loader,
        val_loader=assets.val_loader,
        device=assets.device,
        optimizer=assets.optimizer,
        early_stop_counter=assets.early_stop_counter,
        patience=assets.patience,
        best_val_loss=assets.best_val_loss
    )

    testing(
        model=assets.model,
        criterion=assets.criterion,
        test_loader=assets.test_loader,
        device=assets.device
    )


if __name__ == '__main__':
    main()
    