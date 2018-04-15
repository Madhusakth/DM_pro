from cnn.Setup import Setup

continue_setup = Setup('')
continue_setup.load(rel_filepath='\\setup\\cnn_landmark_32-64-128-256\\setup.json')

X_train_cnn, y_train_one_hot, X_val_cnn, y_val_one_hot, X_test_cnn, y_test_one_hot = continue_setup.getData()

for epoch in range(continue_setup.getEpoch() + 1, 10000):
    print('Epoch %d' % epoch)
    dropout = continue_setup.getModel().fit(X_train_cnn, y_train_one_hot,
                                            batch_size=64, epochs=1, verbose=1,
                                            validation_data=(X_val_cnn, y_val_one_hot))

    continue_setup.updateEpochs(add_epochs=1,
                                train_acc=dropout.history['acc'],
                                train_loss=dropout.history['loss'],
                                val_acc=dropout.history['val_acc'],
                                val_loss=dropout.history['val_loss'],
                                test_acc=[0],
                                test_loss=[0],
                                allow_modify=True)

    continue_setup.save('\\setup\\')
