import argparse


tp = lambda x: list(map(int, x.split('.')))
psr = argparse.ArgumentParser()

psr.add_argument('--address', type=tp)

args = psr.parse_args()

print("address:", args.address)