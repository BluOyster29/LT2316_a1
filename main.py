import get_data, pre_process, config, GRUNetwork
import stats, train, test, argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("-P", "--preset", dest='preset', type=str,
                        help="Choose to use default language set or your own", default="y")
    parser.add_argument("-D", "--Data", dest='data', type=str, default="data/raw/",
                        help="Directory that contains training data")
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
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    x_train, y_train, vocab, int2char, x_test, y_test = get_data.main(args) 
    pre_process.main(args, x_train, y_train, vocab, int2char, x_test, y_test)
    train.main(args)
    test.main()
    
    
    