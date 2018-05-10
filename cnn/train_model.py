from cnn.Setup import Setup
import sys
import keras.backend as K

rel_filepath = sys.argv[1]

continue_setup = Setup('')
continue_setup.load(rel_filepath=rel_filepath)

change_lr = None

if change_lr is not None:
    K.set_value(continue_setup.getModel().optimizer.lr, change_lr)
    print('Changing the model optimizer learning rate to = %f' % K.get_value(continue_setup.getModel().optimizer.lr))
else:
    print('Model optimizer learning rate = %f' % K.get_value(continue_setup.getModel().optimizer.lr))

X_train_cnn, y_train_one_hot, X_val_cnn, y_val_one_hot, X_test_cnn, y_test_one_hot = continue_setup.getData()

for epoch in range(continue_setup.getEpoch() + 1, 10000):
    print('Training \'%s\': Epoch %d' % (continue_setup.getName(), epoch))
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

    continue_setup.save('setup')
