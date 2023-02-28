import os
import sys
import torch
import socket
from datetime import datetime
try:
    from pip._internal.operations import freeze
except ImportError: # pip < 10.0
    from pip.operations import freeze


def get_log_file_name(log_prefix=None):
    log_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs"))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    if log_prefix is not None:
        log_file = f"{log_prefix}"
    else:
        log_file = ""
    date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    log_file = log_file + "__" + date

    n_name_duplicates = len([x for x in os.listdir(log_dir) if x.startswith(log_file)])
    if n_name_duplicates > 0:
        log_file = log_file + "__" + chr(64+n_name_duplicates)
    log_file_name = os.path.join(log_dir, log_file + '_log.txt')
    return log_file_name


class Logger(object):
    def __init__(self, filename, stdout):
        self.terminal = stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

    def _get_writer(self, verbose):
        def write(str):
            if verbose:
                print(str)
            else:
                self.log.write(str+"\n")
        return write

    def log_default(self, args):
        self.log_devices(verbose=True)
        self.log_ptyhon(verbose=True)
        self.log_torch(verbose=True)
        self.log_host(verbose=False)
        self.log_env(verbose=False)
        self.log_args(args, verbose=False)
        print("")

    def log_env(self, verbose=False):
        write = self._get_writer(verbose)
        write("\nEnvironment Info:")
        pkgs = freeze.freeze()
        for pkg in pkgs:
            write(pkg)

    def log_host(self, verbose=False):
        write = self._get_writer(verbose)
        hostname = socket.gethostname()
        write(f"\nHostname: {hostname}")

    def log_ptyhon(self, verbose=False):
        write = self._get_writer(verbose)
        write(f"\nPython Version:\n{sys.version}")

    def log_torch(self, verbose=False):
        write = self._get_writer(verbose)
        write(f"\nTorch Version:\n{torch.__version__}")
        write(f"CUDA Version:\n{torch.version.cuda}")
        write(f"CUDA PATH:\n{os.environ['CUDA_PATH'] if 'CUDA_PATH' in os.environ else 'None'}")
        write(f"CUDA Home:\n{os.environ['CUDA_HOME'] if 'CUDA_HOME' in os.environ else 'None'}")

    def log_devices(self, verbose=False):
        write = self._get_writer(verbose)
        write("\nDevice Info:")
        n_device = torch.cuda.device_count()
        for i in range(n_device):
            write(f"{i}: {torch.cuda.get_device_name(i)}")

    def log_args(self, args, verbose=False):
        write = self._get_writer(verbose)
        write("\nArgs:")

        def write_dict(x, depth=0):
            for key in x.keys():
                if key.startswith("_"): continue
                if "config" in str(type(x[key])):
                    write(f"{depth*'  '}{key}:")
                    write_dict(x[key].__dict__, depth=depth+1)
                    continue
                write(f"{depth*'  '}{key}: {x[key]}")

        write_dict(args.__dict__)
