import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='This is a demo program')
    parser.add_argument('-bs', type=str, required=True, help='Please provide a batch_size')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f'batch size: {args.bs}')

if __name__ == '__main__':
    main()
