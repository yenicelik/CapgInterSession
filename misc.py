def adapt_lr(epoch, para_lr):
    if epoch >= 24:
        learning_rate = para_lr / 1000
    elif epoch >= 16:
        learning_rate = para_lr / 100
    elif epoch >= 9:
        learning_rate = para_lr / 10
    else:
        learning_rate = para_lr

    return learning_rate