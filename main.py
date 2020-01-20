import get_data, pre_process, config, GRUNetwork
import stats, train, test, argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("-P", "--preset", dest='preset', type=str,
                        help="Choose to use default language set or your own", default="y")
    parser.add_argument("-B", "--batch_size", dest='batch_size', type=int, default=1,
                        help="Define the batch size for training")
    parser.add_argument("-F", "--Folder", dest='folder', type=str, default="data/pre_processed/",
                        help="Folder containing outputted csv files")
    parser.add_argument("-E", "--nr_epochs", dest='nr_epochs', type=int,
                        default=2, help="Define the number of training epochs")
    parser.add_argument("-G", "--use_gpu", dest='use_gpu', type=str,
                        help='Use GPU or CPU for training (y/n)', default="y")
    parser.add_argument("-L", "--loss_mode", dest='loss_mode', type=int,
                        help='Choose loss mode', default="1")
    parser.add_argument("-M", "--model_name", dest='model_name', type=str,
                        help='Create name for model')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    x_train, y_train, vocab, int2char, x_test, y_test = pre_process.main(args)
    pre_process.main(args, x_train, y_train, vocab, int2char, x_test, y_test)
    trained_model = train.main(args)
    test.main(trained_model)
