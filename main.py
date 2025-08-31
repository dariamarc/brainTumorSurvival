from data_generator import MRIDataGenerator
from losses import FocalLoss
from tensorflow import keras

from model import MProtoNet3D_Segmentation_Keras

if __name__ == "__main__":
    folder_path = "archive/BraTS2020_training_data/content/data"
    batch_size = 1
    split_ratio = 0.2
    random_state = 42

    D = 155
    H = 240
    W = 240
    C = 4

    input_shape = (D, H, W, C)

    # no of output classes: GD enhancing tumor, peritumoral edema, non-enhancing tumor core
    num_output_classes = 3

    train_generator = MRIDataGenerator(
        folder_path,
        batch_size=batch_size,
        num_slices=D,
        num_volumes=369,
        split_ratio=split_ratio,
        subset='train',
        shuffle=True,
        random_state=random_state
    )
    validation_generator = MRIDataGenerator(
        folder_path,
        batch_size=batch_size,
        num_slices=D,
        num_volumes=369,
        split_ratio=split_ratio,
        subset='val',
        shuffle=False,
        random_state=random_state
    )

    # the first number has to divide with the number of output classes
    # 21 / 3 = 7 - we will have 7 prototypes learned for each class
    prototype_shape = (21, 128, 1, 1, 1)

    model = MProtoNet3D_Segmentation_Keras(
        in_size=input_shape,
        num_classes=num_output_classes,
        prototype_shape=prototype_shape,
        features='resnet50_ri',
        f_dist='l2'
    )

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    focal_loss_fn = FocalLoss(gamma=2.0)


    def combined_segmentation_loss(y_true, y_pred):
        classification_loss = focal_loss_fn(y_true, y_pred)
        return classification_loss


    model.compile(optimizer=optimizer,
                  loss=combined_segmentation_loss,
                  metrics=['accuracy'])

    epochs = 1
    steps_per_epoch = None
    validation_steps = None

    print("Starting model training...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        verbose=2
    )
    print("Training finished.")

    # You can now save the model or make predictions
    # model.save('my_3d_segmentation_model.keras')