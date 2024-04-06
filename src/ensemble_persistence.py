from models import PersistenceEnsemble
import visualization as viz


if __name__ == "__main__":

    NUM_QUANTILES = 10
    BATCH_SIZE = 1

    peen = PersistenceEnsemble(n_quantiles=NUM_QUANTILES)
    peen.create_dataloaders(
        path="datasets/moving_mnist_dataset/", batch_size=BATCH_SIZE
    )

    in_frames, out_frames, predictions, crps = peen.random_example()

    viz.show_image_list(
        images_list=in_frames[0].tolist(),
        fig_name="figures/peen_input.jpg",
        save_fig=True,
    )

    viz.show_image_list(
        images_list=predictions[0].tolist(),
        fig_name="figures/peen_quantiles_pred.jpg",
        save_fig=True,
    )
